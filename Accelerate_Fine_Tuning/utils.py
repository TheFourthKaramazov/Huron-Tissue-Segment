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


# Helper functions
def load_hyperparameters(yaml_file):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Arguments to fine-tune Mask2Former")
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=False)
    parser.add_argument("--experiment_name", type=str, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--freeze_classifier", type=str, required=False)
    parser.add_argument("--freeze_pixel_module", type=str, required=False)
    parser.add_argument("--freeze_transformer_decoder", type=str, required=False)
    parser.add_argument("--lr", type=float, required=False)
    parser.add_argument("--num_labels", type=int, required=False)

    args = parser.parse_args()

    # Load hyperparameters from YAML file
    with open(yaml_file, 'r') as file:
        hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

    # Override hyperparameters from command-line arguments
    for key, value in hyperparameters.items():
        if key in args.__dict__ and args.__dict__[key] is not None:
            hyperparameters[key] = args.__dict__[key]

    return hyperparameters


def freeze_and_prepare(model, accelerator, hparams):
    # Freeze some layers in the model if wanted
    for name, param in model.named_parameters():
        if hparams["freeze_classifier"] == "True":
            if "class_predictor" in name:
                param.requires_grad = False
        elif hparams["freeze_pixel_module"] == "True":
            if "pixel_level_module" in name:
                param.requires_grad = False
        elif hparams["freeze_transformer_decoder"] == "True":
            if "transformer_module.decoder" in name:
                param.requires_grad = False
        else:
            param.requires_grad = True

    # Display the trainable layers for confirmation
    if accelerator.is_main_process:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        total_params = sum(p.numel() for p in model.parameters())

        print(f"Total parameters: {total_params}. Trainable parameters: {num_trainable_params}")
        print(f"Trainable parameters represent {(num_trainable_params / total_params) * 100}% of total")


def setup_experiment_directories(experiment_dir, accelerator, hparam_file):
    logs_path = os.path.join(experiment_dir, "training.log")

    # Create experiment folder
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        if not os.path.exists(logs_path):
            with open(logs_path, "w") as f:
                f.write(" ")
        print(f"Results and checkpoints will be saved to {experiment_dir}")

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler(logs_path)])

    # Copy hparams to project dir for future reference
    shutil.copy(hparam_file, experiment_dir)


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
