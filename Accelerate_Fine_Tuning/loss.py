import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import conv2d


def should_return_low_loss(inputs, targets, threshold_zero_loss):
    """Helper function for loss"""
    
    return (inputs.sum(dim=(1, 2, 3)) <= threshold_zero_loss) & (targets.sum(dim=(1, 2, 3)) <= threshold_zero_loss)


def compute_low_loss(inputs, targets, threshold_zero_loss):
    """ Helper function for loss"""
    return 1 / 2 ** ((2.75 * threshold_zero_loss - inputs.sum(dim=(1, 2, 3)) - targets.sum(dim=(1, 2, 3))))


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks. Measures the overlap between
    predicted and ground truth masks, emphasizing small structures.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate Dice score
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BoundaryDiceLoss(nn.Module):
    """
    Dice loss paired with boundary loss
    """

    def __init__(self, smooth=1, boundary_weight=0.5, threshold_zero_loss=10):
        """
        Balanced means that the weights is calculated using the
        proportion of pixels of each type in the predicted image
        """
        super(BoundaryDiceLoss, self).__init__()
        self.tissueDice = DiceLoss(smooth)
        self.boundaryDice = DiceLoss(smooth)
        self.threshold_zero_loss = threshold_zero_loss
        self.boundary_weight = boundary_weight

    def boundary_loss(self, predictions, targets):
        kernel = torch.ones((1, 1, 3, 3), device=predictions.device)
        pred_boundaries = conv2d(predictions, kernel, padding=1).clamp(0, 1) - predictions
        target_boundaries = conv2d(targets, kernel, padding=1).clamp(0, 1) - targets
        return pred_boundaries, target_boundaries

    def forward(self, inputs, targets):
        mask_of_low_loss = should_return_low_loss(inputs, targets, self.threshold_zero_loss)

        inputs_low_loss = inputs[mask_of_low_loss]
        targets_low_loss = targets[mask_of_low_loss]

        if inputs_low_loss.numel() > 0:
            low_losses = compute_low_loss(inputs_low_loss, targets_low_loss, self.threshold_zero_loss)
        else:
            low_losses = torch.tensor(0.0, device=inputs.device)

        inputs_normal_loss = inputs[~mask_of_low_loss]
        targets_normal_loss = targets[~mask_of_low_loss]

        if inputs_normal_loss.numel() > 0:
            normal_losses = self.tissueDice(inputs_normal_loss,
                                            targets_normal_loss) + self.boundary_weight * self.boundaryDice(
                *self.boundary_loss(inputs_normal_loss, targets_normal_loss))
        else:
            normal_losses = torch.tensor(0.0, device=inputs.device)

        return (
            ~mask_of_low_loss).float().mean() * normal_losses.mean() + mask_of_low_loss.float().mean() * low_losses.mean()


class ScaledDiceLoss(nn.Module):
    """
    Scaled version of the dice loss
    """

    def __init__(self, smooth=1, threshold_zero_loss=10):
        super(ScaledDiceLoss, self).__init__()
        self.tissueDice = DiceLoss(smooth)
        self.threshold_zero_loss = threshold_zero_loss

    def forward(self, inputs, targets):
        mask_of_low_loss = should_return_low_loss(inputs, targets, self.threshold_zero_loss)

        inputs_low_loss = inputs[mask_of_low_loss]
        targets_low_loss = targets[mask_of_low_loss]

        if inputs_low_loss.numel() > 0:
            low_losses = compute_low_loss(inputs_low_loss, targets_low_loss, self.threshold_zero_loss)
        else:
            low_losses = torch.tensor(0.0, device=inputs.device)

        inputs_normal_loss = inputs[~mask_of_low_loss]
        targets_normal_loss = targets[~mask_of_low_loss]

        if inputs_normal_loss.numel() > 0:
            scaling_factor = torch.log(
                (inputs_normal_loss.sum() + targets_normal_loss.sum()) / inputs_normal_loss.size(0))
            normal_losses = scaling_factor * self.tissueDice(inputs_normal_loss, targets_normal_loss)
        else:
            normal_losses = torch.tensor(0.0, device=inputs.device)

        return (
            ~mask_of_low_loss).float().mean() * normal_losses.mean() + mask_of_low_loss.float().mean() * low_losses.mean()


def calculate_iou_infer(pred, target):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Treat cases with zero union as perfect matches.

    Args:
        pred (numpy.ndarray): Predicted binary mask.
        target (numpy.ndarray): Ground truth binary mask.

    Returns:
        float: IoU score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    if union == 0:  # If union is zero, treat as a perfect match
        return 1.0

    return intersection / (union + 1e-6)  # Calculate IoU


def calculate_dice_infer(pred, target):
    """
    Calculate Dice coefficient between two binary masks.
    Treat cases with zero denominator as perfect matches.

    Args:
        pred (numpy.ndarray): Predicted binary mask.
        target (numpy.ndarray): Ground truth binary mask.

    Returns:
        float: Dice coefficient score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    denominator = pred.sum() + target.sum()

    if denominator == 0:  # If denominator is zero, treat as a perfect match
        return 1.0

    return (2 * intersection) / (denominator + 1e-6)  # Calculate Dice


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Dice and Binary Cross-Entropy (BCE) Loss.

    Args:
        dice_weight (float): Weight for Dice Loss.
        bce_weight (float): Weight for BCE Loss.
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(CombinedDiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        ## Calculate Dice loss
        dice_loss = self.dice(logits, targets)

        ## Calculate BCE loss
        bce_loss = self.bce(logits, targets)

        # Combine Dice and BCE losses
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class DoubleDiceLoss(nn.Module):
    """
    Combination of the dice loss of the background and tissue.
    """

    def __init__(self, smooth=1, weight_tissue='balanced'):
        """
        Balanced means that the weights is calculated using the
        proportion of pixels of each type in the predicted image
        """
        super(DoubleDiceLoss, self).__init__()
        self.tissueDice = DiceLoss(smooth)
        self.backgroundDice = DiceLoss(smooth)
        self.weight_tissue = weight_tissue

    def forward(self, inputs, targets):
        if self.weight_tissue == 'balanced':
            num_ones = targets.sum()
            total_pixels = targets.numel()
            weight_tissue = num_ones.float() / total_pixels
        else:
            weight_tissue = self.weight_tissue

        inverted_inputs = (-inputs + inputs.max())
        inverted_masks = (-targets + inputs.max())

        return weight_tissue * self.tissueDice(inputs, targets) + (1 - weight_tissue) * self.tissueDice(inverted_inputs,
                                                                                                        inverted_masks)