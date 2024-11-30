from tqdm import tqdm
import os
import shutil
from utils import *
import torch.nn.functional as F
import logging
import torch


def train(accelerator, model, train_loader, val_loader, criterion, optimizer, epochs, experiment_dir):
    model.train()
    best_iou = 0.0

    for epoch in range(epochs):
        running_loss = 0.0

        for pixel_values, masks in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training",
                                        disable=not accelerator.is_main_process):
            outputs = model(pixel_values=pixel_values.float())
            tissue_logits = outputs.masks_queries_logits[:, 1]

            # Resize logits to match masks
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),  # Add channel dimension
                size=masks.shape[-2:],  # Match mask size
                mode="bilinear",
                align_corners=False
            ))

            # Zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = criterion(tissue_logits_resized, masks)

            # Use accelerator for backward pass
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Run validation
        avg_val_loss, avg_iou, avg_dice = validate(accelerator, model, val_loader, criterion)

        # Checkpointing and logging
        if accelerator.is_main_process:
            logging.info(
                f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss:.4f},"
                f" Validation Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}")

            best_iou = save_checkpoint(accelerator, model, experiment_dir, avg_iou, best_iou)


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
    num_samples = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):
            # Process inputs and get model outputs
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

    avg_val_loss = val_loss / len(val_loader)
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0

    return avg_val_loss, avg_iou, avg_dice


def test(accelerator, model, test_loader, criterion):
    """
    Tester function aligned with inference logic, including IoU and Dice metric calculation.

    Args:
        model: The trained segmentation model.
        test_loader: DataLoader providing testing images and ground truth masks.
        criterion: Loss function for evaluation.

    Returns:
        avg_test_loss: Average test loss.
        avg_iou: Average IoU across the test set.
        avg_dice: Average Dice score across the test set.
    """
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(test_loader, desc="Testing", disable=not accelerator.is_main_process):
            # Process inputs and get model outputs
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

    avg_test_loss = val_loss / len(test_loader)
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0

    # Save metrics to logs
    if accelerator.is_main_process:
        logging.info(f"Test Loss: {avg_test_loss:.4f}, Test IoU: {avg_iou:.4f}, Test Dice: {avg_dice:.4f}")


def save_checkpoint(accelerator, model, experiment_dir, iou, best_iou):
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
