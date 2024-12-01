# System libraries
import os
import sys
import shutil
import logging

# Deep learning libraries
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Project libraries
sys.path.append('..')
from loss import calculate_dice_infer, calculate_iou_infer


def calculate_metrics(output, target):
    """
    Compute IoU, Dice, and pixel accuracy for given outputs and ground truth masks.

    Args:
        output: Predicted segmentation logits (after resizing and applying activation).
        target: Ground truth segmentation masks.

    Returns:
        avg_iou: Average Intersection over Union (IoU) for the batch.
        avg_dice: Average Dice coefficient for the batch.
        pixel_accuracy: Pixel accuracy across all samples in the batch.
    """
    predicted_masks = (output > 0.5).cpu().numpy().astype(np.uint8)
    ground_truth_masks_np = target.cpu().numpy().astype(np.uint8)

    # Compute metrics
    num_samples = 0
    total_iou = 0
    total_dice = 0
    for pred, gt in zip(predicted_masks, ground_truth_masks_np):
        total_iou += calculate_iou_infer(pred, gt)
        total_dice += calculate_dice_infer(pred, gt)
        num_samples += 1

    # Calculate pixel accuracy
    matching_pixels = torch.tensor(predicted_masks == ground_truth_masks_np).sum()  # Count matching pixels per batch
    total_pixels = torch.tensor(predicted_masks).numel()  # Total number of pixels per batch
    pixel_accuracy = float(matching_pixels) / total_pixels

    # Calculate average IoU and Dice metrics
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0

    return avg_iou, avg_dice, pixel_accuracy


def train(accelerator, model, train_loader, val_loader, criterion, optimizer, epochs, experiment_dir):
    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_pixel_acc = 0.0

        for pixel_values, masks in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training",
                                        disable=not accelerator.is_main_process):

            # Pass through model and extract tissue logits
            outputs = model(pixel_values=pixel_values.float())
            tissue_logits = outputs.masks_queries_logits[:, 1]

            # Resize logits to match masks
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),  # Add channel dimension
                size=masks.shape[-2:],  # Match mask size
                mode="bilinear",
                align_corners=False
            ))

            # Compute loss
            loss = criterion(tissue_logits_resized, masks)

            # Zero gradients
            optimizer.zero_grad()
            # Use accelerator for backward pass
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()
            iou, dice, pixel_accuracy = calculate_metrics(tissue_logits_resized, masks)
            running_iou += iou
            running_dice += dice
            running_pixel_acc += pixel_accuracy

        # Average training metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_train_iou = running_iou / len(train_loader)
        avg_train_dice = running_dice / len(train_loader)
        avg_train_pixel_accuracy = running_pixel_acc / len(train_loader)

        # Run validation and compute validation metrics
        avg_val_loss, avg_val_iou, avg_val_dice, avg_val_pixel_accuracy = validate(
            accelerator, model, val_loader, criterion)

        # Checkpointing and logging
        if accelerator.is_main_process:
            logging.info(
                f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.4f},"
                f" IoU: {avg_train_iou:.4f}, Dice: {avg_train_dice:.4f},"
                f" Pixel Accuracy: {avg_train_pixel_accuracy:.4f}.")
            logging.info(f"Epoch [{epoch + 1}/{epochs}],"
                         f" Validation Loss: {avg_val_loss:.4f}, IoU: {avg_val_iou:.4f},"
                         f" Dice: {avg_val_dice:.4f}, Pixel Accuracy: {avg_val_pixel_accuracy:.4f}")

            best_iou = save_best_checkpoint(accelerator, model, experiment_dir, avg_val_iou, best_iou)


def validate(accelerator, model, val_loader, criterion):
    """
    Validation function aligned with inference logic, including IoU and Dice metric calculation.

    Args:
        accelerator: The accelerator object.
        model: The trained segmentation model.
        val_loader: DataLoader providing validation images and ground truth masks.
        criterion: Loss function for evaluation.

    Returns:
        avg_val_loss: Average validation loss.
        avg_iou: Average IoU across the validation set.
        avg_dice: Average Dice score across the validation set.
    """
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):

            # Pass through model and extract tissue logits
            outputs = model(pixel_values=images.float())
            tissue_logits = outputs.masks_queries_logits[:, 1]

            # Resize logits to match ground truth mask size
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),  # Add channel dimension
                size=ground_truth_masks.shape[-2:],  # Match mask size
                mode="bilinear",
                align_corners=False
            ))

            # Compute loss
            loss = criterion(tissue_logits_resized, ground_truth_masks)
            val_loss += loss.item()

            # Convert predicted logits to binary masks
            predicted_masks = (tissue_logits_resized > 0.5).cpu().numpy().astype(np.uint8)
            ground_truth_masks_np = ground_truth_masks.cpu().numpy().astype(np.uint8)

            # Compute metrics
            for pred, gt in zip(predicted_masks, ground_truth_masks_np):
                total_iou += calculate_iou_infer(pred, gt)
                total_dice += calculate_dice_infer(pred, gt)
                num_samples += 1

            matching_pixels = (predicted_masks == ground_truth_masks_np).sum()  # Count matching pixels per batch
            total_pixels = torch.tensor(predicted_masks).numel()
            total_pixel_acc += float(matching_pixels) / total_pixels

    avg_val_loss = val_loss / len(val_loader)
    avg_val_pixel_acc = total_pixel_acc / len(val_loader)
    avg_val_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_val_dice = total_dice / num_samples if num_samples > 0 else 0

    return avg_val_loss, avg_val_iou, avg_val_dice, avg_val_pixel_acc


def save_best_checkpoint(accelerator, model, experiment_dir, iou, best_iou):
    # Save new checkpoint when IOU increases
    if iou > best_iou:
        for file in os.listdir(experiment_dir):
            if file == "best_iou_checkpoint.pt":
                shutil.rmtree(os.path.join(experiment_dir, file))

        best_iou = iou
        checkpoint_path = os.path.join(experiment_dir, f"best_iou_checkpoint.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_path)
        logging.info(f"IOU has increased to {best_iou:.4f}, saved new model checkpoint.")

    return best_iou
