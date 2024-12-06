import os
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


# Utility functions
def crop_black_borders(image, threshold=30):
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


def gaussian_blur(image, radius=2):
    """Apply Gaussian blur to an image to avoid capturing noise as tissue."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def preprocess_image(image_path, target_size=(128, 128)):
    """Preprocess an image: crop black borders, enhance contrast, and resize."""
    image = Image.open(image_path).convert("RGB")
    # cropped_image = crop_black_borders(image) # cropping causes probllems

    # Blur to remove noise
    blurred_image = gaussian_blur(image, 2)  # Increase radius if desired for more blur

    # Resize to the target size
    resized_image = blurred_image.resize(target_size, Image.BICUBIC)

    return resized_image


def preprocess_mask(mask_path, target_size=(128, 128)):
    """Convert mask to binary, ensure tissue is white and background is black, and resize."""
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    mask_array = np.array(mask)

    # Apply binary threshold and ensure tissue is white, background is black
    binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)  # Normalize mask to [0, 1]
    binary_mask = Image.fromarray(binary_mask * 255)  # Convert back to PIL Image

    # Resize to the target size using nearest-neighbor interpolation
    resized_mask = binary_mask.resize(target_size, Image.NEAREST)

    return resized_mask


# Dataset class
class TissueDataset(Dataset):
    def __init__(self, image_files, mask_files, image_processor=None, mask_transform=None):
        """
        Initialize the TissueDataset.

        Parameters:
        - image_files: List of image file paths.
        - mask_files: List of mask file paths.
        - image_processor: Preprocessing function for images (expects PIL input).
        - mask_transform: Preprocessing function for masks.
        """
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_processor = image_processor
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask as PIL images
        image = Image.open(self.image_files[idx]).convert("RGB")  # Convert to PIL RGB
        mask = Image.open(self.mask_files[idx]).convert("L")  # Convert to PIL Grayscale

        # Process images
        image = preprocess_image(self.image_files[idx])
        mask = preprocess_mask(self.mask_files[idx])

        # Process image using the image processor
        if self.image_processor:
            image = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        # Process the mask using mask_transform (if provided)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


# Data preparation functions
def get_file_paths(image_folder, mask_folder):
    """Retrieve sorted lists of image and mask file paths."""
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder))]
    mask_files = [os.path.join(mask_folder, f) for f in sorted(os.listdir(mask_folder))]
    assert len(image_files) == len(mask_files), "Mismatch between image and mask files."
    return image_files, mask_files


# Define transforms for images and masks
image_transform = transforms.Compose([

    transforms.ToTensor(),  # Convert to tensor
])

mask_transform = transforms.Compose([

    transforms.ToTensor(),  # Convert to tensor
])


def create_dataloaders(image_folder, mask_folder, batch_size=16, image_processor=None, mask_transform=mask_transform,
                       divide=1):
    """
    Create train and validation DataLoaders.

    Parameters:
    - image_folder: Path to the image folder.
    - mask_folder: Path to the mask folder.
    - batch_size: Batch size for DataLoaders.
    - image_processor: Optional image processor for preprocessing.
    - mask_transform: Optional transform for masks.
    - divide: Divide the dataset by this number. Default is 1.

    Returns:
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    """

    # Get image and mask file paths
    image_files, mask_files = get_file_paths(image_folder, mask_folder)

    # Create dataset
    dataset = TissueDataset(
        image_files=image_files,
        mask_files=mask_files,
        image_processor=image_processor,  # Pass the image processor here
        mask_transform=mask_transform  # Pass the mask transform here
    )

    # Cut dataset for testing (optional for quick parameter testing)
    dataset = torch.utils.data.Subset(dataset, range(0, len(dataset) // divide))

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, dataset, image_files, mask_files
