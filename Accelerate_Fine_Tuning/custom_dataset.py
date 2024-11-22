import os
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


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

def preprocess_image(image_path):
    """Preprocess an image: crop black borders and enhance contrast."""
    image = Image.open(image_path).convert("RGB")
    cropped_image = crop_black_borders(image)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(cropped_image)
    enhanced_image = enhancer.enhance(10)

    return enhanced_image

def preprocess_mask(mask_path):
    """Convert mask to binary and ensure tissue is white and background is black."""
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    mask_array = np.array(mask)

    # Apply binary threshold and ensure tissue is white, background is black
    binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)  # Normalize mask to [0, 1]
    return Image.fromarray(binary_mask * 255)  # Return a PIL Image with values 0 or 255

class TissueDataset(Dataset):
    def __init__(self, image_files, mask_files, image_folder, mask_folder, transform=None, mask_transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_files[idx])

        image = preprocess_image(img_path)
        mask = preprocess_mask(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

