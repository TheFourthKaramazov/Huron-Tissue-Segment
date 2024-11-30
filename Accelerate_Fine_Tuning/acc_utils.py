from PIL import Image
import numpy as np
import torch.nn as nn
import os
import logging
import argparse
import yaml
import torch
import shutil


# Helper classes
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Apply sigmoid to inputs if not already done
        inputs = torch.sigmoid(inputs)

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class CombinedDiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(CombinedDiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Dice Loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2))  # Sum over spatial dimensions only
        dice_loss = 1 - (2. * intersection + self.smooth) / (probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + self.smooth)
        dice_loss = dice_loss.mean()  # Average over the batch

        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Combine losses
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# Helper function
def calculate_iou_infer(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Works with NumPy arrays.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)  # Avoid division by zero


def calculate_dice_infer(pred, target):
    """
    Calculate Dice coefficient between two binary masks.
    Works with NumPy arrays.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    return (2 * intersection) / (pred.sum() + target.sum() + 1e-6)  # Avoid division by zero


def crop_black_borders(image, threshold):
    """Crop black borders from an image based on a threshold for black pixels."""
    img_array = np.array(image)
    gray_img = np.mean(img_array, axis=2)  # Convert to grayscale by averaging channels

    # Initialize cropping boundaries
    top, bottom = 0, gray_img.shape[0]
    left, right = 0, gray_img.shape[1]

    # Crop from the top
    while top < bottom and np.mean(gray_img[top, :]) <= threshold:
        top += 1

    # Crop from the bottom
    while bottom > top and np.mean(gray_img[bottom - 1, :]) <= threshold:
        bottom -= 1

    # Crop from the left
    while left < right and np.mean(gray_img[:, left]) <= threshold:
        left += 1

    # Crop from the right
    while right > left and np.mean(gray_img[:, right - 1]) <= threshold:
        right -= 1

    # Crop the image to the calculated bounds
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def preprocess_image(image_path, target_size=(128,128), threshold=30):
    # Crop black borders
    image = Image.open(image_path).convert("RGB")
    #cropped_image = crop_black_borders(image, threshold)

    # Resize to target size
    resized_image = image.resize(target_size, Image.BICUBIC)

    return resized_image


def preprocess_mask(mask_path, target_size=(128,128)):
    """Convert mask to binary and ensure tissue is white and background is black."""
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    mask_array = np.array(mask)

    # Apply binary threshold and ensure tissue is white, background is black
    binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)  # Normalize mask to [0, 1]
    binary_mask = Image.fromarray(binary_mask * 255)  # Return a PIL Image with values 0 or 255

    # Resize
    resized_mask = binary_mask.resize(target_size, Image.NEAREST)

    return resized_mask


def resize_mask(mask, target_shape=(512, 512)):
    """Resize a mask to the target shape using nearest neighbor interpolation."""
    mask = np.squeeze(mask).astype(np.uint8)
    mask = Image.fromarray(mask).convert("L")
    resized_mask = mask.resize(target_shape, Image.NEAREST)
    return np.array(resized_mask)


def invert_mask(mask):
    """Invert binary mask (0 becomes 1 and vice versa)."""
    return np.where(mask == 0, 1, 0).astype(np.uint8)


# Metric functions
def iou_score(pred, target):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 1.0


def dice_score(pred, target):
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    intersection = np.sum(pred * target)
    return (2. * intersection) / (np.sum(pred) + np.sum(target)) if (np.sum(pred) + np.sum(target) > 0) else 1.0
