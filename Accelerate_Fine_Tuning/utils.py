from PIL import Image
import numpy as np
import torch.nn as nn
import torch


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
