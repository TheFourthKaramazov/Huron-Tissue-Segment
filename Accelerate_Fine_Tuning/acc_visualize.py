# Deep learning librairies
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
# System librairies
import os
# Project classes/functions
from acc_custom_dataset import TissueDataset
from transformers import Mask2FormerForUniversalSegmentation
import torch.nn.functional as F
from acc_model import create_mask2former

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


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


def infer_and_display(model, image_processor, dataloader, device, num_samples=5, target_size=(128, 128)):
    """
    Perform inference and display the original image, ground truth mask, and predicted mask side by side.

    Args:
        model: The trained segmentation model.
        image_processor: Preprocessing module for the input images.
        dataloader: DataLoader providing images and ground truth masks.
        device: Computation device (CPU or CUDA).
        num_samples: Number of samples to visualize.
        target_size: Target size for resizing masks during visualization.
    """
    model.eval()
    samples_displayed = 0
    total_iou = 0.0
    total_dice = 0.0
    num_evaluated = 0

    with torch.no_grad():
        for images, ground_truth_masks in tqdm(dataloader, desc="Inferencing"):
            # Move ground truth masks to device and convert to float
            ground_truth_masks = ground_truth_masks.to(device, dtype=torch.float32)

            # Convert tensors to PIL images for the image processor
            pil_images = [transforms.ToPILImage()(img) for img in images]
            inputs = image_processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            tissue_logits = outputs.masks_queries_logits[:, 1]  # Binary segmentation logits

            # Resize logits to match mask size
            tissue_logits_resized = torch.sigmoid(F.interpolate(
                tissue_logits.unsqueeze(1),
                size=ground_truth_masks.shape[-2:],  # Match ground truth mask size
                mode="bilinear",
                align_corners=False
            ).squeeze(1))  # Remove channel dimension

            # Convert predicted logits to binary masks
            predicted_masks = (tissue_logits_resized > 0.5).cpu().numpy().astype(np.uint8)

            # Display the first few samples
            for i in range(len(images)):
                if samples_displayed >= num_samples:
                    # Print the average metrics
                    avg_iou = total_iou / num_evaluated
                    avg_dice = total_dice / num_evaluated
                    print(f"\nMean IoU: {avg_iou:.4f}")
                    print(f"\nMean Dice: {avg_dice:.4f}")
                    return

                # Convert images, ground truths, and predictions to displayable formats
                original_image = pil_images[i]
                ground_truth_mask = ground_truth_masks[i].cpu().numpy().squeeze()  # Remove extra dimensions
                predicted_mask = predicted_masks[i]

                # Ensure ground truth mask is binary (0 or 1) and scaled to 255
                ground_truth_mask = (ground_truth_mask > 0.5).astype(np.uint8) * 255

                # Calculate IoU and Dice for the current sample
                iou = calculate_iou_infer(predicted_mask, ground_truth_mask // 255)  # Divide by 255 for binary comparison
                dice = calculate_dice_infer(predicted_mask, ground_truth_mask // 255)  # Divide by 255 for binary comparison
                total_iou += iou
                total_dice += dice
                num_evaluated += 1

                # Resize ground truth and predicted masks for visualization
                ground_truth_mask_resized = np.array(Image.fromarray(ground_truth_mask).resize(target_size, Image.NEAREST))
                predicted_mask_resized = np.array(Image.fromarray((predicted_mask * 255).astype(np.uint8)).resize(target_size, Image.NEAREST))

                # Display side by side
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(original_image)
                axs[0].set_title(f"Original Image {samples_displayed + 1}")
                axs[0].axis("off")

                axs[1].imshow(ground_truth_mask_resized, cmap="gray", vmin=0, vmax=255)
                axs[1].set_title(f"Ground Truth Mask {samples_displayed + 1}")
                axs[1].axis("off")

                axs[2].imshow(predicted_mask_resized, cmap="gray", vmin=0, vmax=255)
                axs[2].set_title(f"Predicted Mask {samples_displayed + 1}")
                axs[2].axis("off")

                plt.show()
                samples_displayed += 1

    # Print the final average metrics
    avg_iou = total_iou / num_evaluated if num_evaluated > 0 else 0
    avg_dice = total_dice / num_evaluated if num_evaluated > 0 else 0
    print(f"\nMean IoU: {avg_iou:.4f}")
    print(f"Mean Dice: {avg_dice:.4f}")


# load the fine-tuned model
model = Mask2FormerForUniversalSegmentation.from_pretrained('all_models\\100-epochs-unfrozen\\100_epochs\\checkpoints\\checkpoint-epoch-95')
image_processor, _ = create_mask2former()

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create inference loader
# Define paths
image_folder = os.path.join("Huron_data", "Sliced_Images")
mask_folder = os.path.join("Huron_data", "Sliced_masks")

# Get sorted lists of image and mask files
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Mask transform - avoid normalization if not needed
mask_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

# Create dataset and dataloaders
dataset = TissueDataset(
    image_files, mask_files,
    image_folder, mask_folder,
    transform=image_transform,
    mask_transform=mask_transform
)

# Take subset of dataset for inference
infer_size = int(0.01 * len(dataset))
leftover_size = len(dataset) - infer_size
print(f"Infer_size: {infer_size}")
infer_dataset, _ = torch.utils.data.random_split(dataset, [infer_size, leftover_size])
infer_loader = DataLoader(infer_dataset, batch_size=32, shuffle=False)

# Perform inference
infer_and_display(model, image_processor, infer_loader, device, num_samples=5)
