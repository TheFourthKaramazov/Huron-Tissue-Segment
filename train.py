import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from loss import *


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


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device="cuda"):
    """
    Train the segmentation model and evaluate metrics after each epoch.

    Args:
        model: The segmentation model to train.
        train_loader: DataLoader providing training images and masks.
        val_loader: DataLoader providing validation images and masks.
        criterion: Loss function for training and validation.
        optimizer: Optimizer for updating model weights.
        num_epochs: Number of training epochs.
        device: Computation device (CPU or CUDA).

    Returns:
        train_metrics: Dictionary containing training loss, IoU, Dice, and pixel accuracy per epoch.
        val_metrics: Dictionary containing validation loss, IoU, Dice, and pixel accuracy per epoch.
    """
    train_metrics = {'loss': [], 'iou': [], 'dice': [], 'pixel_acc': []}
    val_metrics = {'loss': [], 'iou': [], 'dice': [], 'pixel_acc': []}
    model.to(device)
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_pixel_acc = 0.0

        for pixel_values, masks in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training"):
            # Move inputs and masks to the correct device
            pixel_values = pixel_values.to(device, dtype=next(model.parameters()).dtype)  # Match model dtype
            masks = masks.to(device, dtype=torch.float32)  # Ensure masks are on the correct device and float32

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            tissue_logits = outputs.masks_queries_logits[:, 1]  # Binary segmentation logits

            # Resize logits to match masks
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),  # Add channel dimension
                size=masks.shape[-2:],  # Match mask size
                mode="bilinear",
                align_corners=False
            ))  # Keep channel dimension for metrics

            # Compute loss
            loss = criterion(tissue_logits_resized, masks)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and metrics
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
        train_metrics['loss'].append(avg_train_loss)
        train_metrics['iou'].append(avg_train_iou)
        train_metrics['dice'].append(avg_train_dice)
        train_metrics['pixel_acc'].append(avg_train_pixel_accuracy)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training IoU: {avg_train_iou:.4f}, Training Dice: {avg_train_dice:.4f}, Training Pixel Acc: {avg_train_pixel_accuracy:.4f}")

        # Perform validation
        avg_val_loss, avg_iou, avg_dice, avg_pixel_acc = validate(model, val_loader, criterion, device)
        val_metrics['loss'].append(avg_val_loss)
        val_metrics['iou'].append(avg_iou)
        val_metrics['dice'].append(avg_dice)
        val_metrics['pixel_acc'].append(avg_pixel_acc)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Validation - Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Dice: {avg_dice:.4f}, Pixel acc: {avg_pixel_acc:.4f}\n")

        # Stop criteria
        if avg_train_iou < 0.3:
            print("Stopping criteria: not enough potential!")
            return train_metrics, val_metrics

    return train_metrics, val_metrics


def validate(model, val_loader, criterion, device):
    """
    Validate the segmentation model and compute evaluation metrics.

    Args:
        model: The trained segmentation model.
        val_loader: DataLoader providing validation images and ground truth masks.
        criterion: Loss function for evaluation.
        device: Computation device (CPU or CUDA).

    Returns:
        avg_val_loss: Average validation loss across the validation set.
        avg_iou: Average Intersection over Union (IoU) across the validation set.
        avg_dice: Average Dice coefficient across the validation set.
        avg_pixel_acc: Average pixel accuracy across the validation set.
    """
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(val_loader, desc="Validation"):
            # Move inputs and masks to the correct device
            images = images.to(device)
            ground_truth_masks = ground_truth_masks.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(pixel_values=images)
            tissue_logits = outputs.masks_queries_logits[:, 1]  # Binary segmentation logits

            # Resize logits to match mask size
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),  # Add channel dimension
                size=ground_truth_masks.shape[-2:],  # Match mask size
                mode="bilinear",
                align_corners=False
            ))  # Keep channel dimension for metrics

            # Compute loss
            loss = criterion(tissue_logits_resized, ground_truth_masks)
            val_loss += loss.item()

            # Calculate metrics
            predicted_masks = (tissue_logits_resized > 0.5).cpu().numpy().astype(np.uint8)
            ground_truth_masks_np = ground_truth_masks.cpu().numpy().astype(np.uint8)

            for pred, gt in zip(predicted_masks, ground_truth_masks_np):
                total_iou += calculate_iou_infer(pred, gt)
                total_dice += calculate_dice_infer(pred, gt)
                num_samples += 1

            matching_pixels = (predicted_masks == ground_truth_masks_np).sum()  # Count matching pixels per batch
            total_pixels = torch.tensor(predicted_masks).numel()
            total_pixel_acc += float(matching_pixels) / total_pixels

    # Calculate average metrics
    avg_val_loss = val_loss / len(val_loader)
    avg_val_pixel_acc = total_pixel_acc / len(val_loader)
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0

    return avg_val_loss, avg_iou, avg_dice, avg_val_pixel_acc