import os
from torch.utils.data import Dataset, DataLoader
from utils import preprocess_image, preprocess_mask
import torch
from torchvision import transforms


class TissueDataset(Dataset):
    def __init__(self, image_files, mask_files, augmentations=None, image_processor=None, border_threshold=30):
        self.image_files = image_files
        self.mask_files = mask_files
        self.augmentations = augmentations
        self.image_processor = image_processor
        self.border_threshold = border_threshold

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # pre-process images
        image = preprocess_image(self.image_files[idx], threshold=self.border_threshold)
        mask = preprocess_mask(self.mask_files[idx])

        # augment if specified
        if self.augmentations:
            image = self.augmentations(image)
            mask = self.augmentations(mask)
        else:
            mask_transform = transforms.Compose([transforms.ToTensor()])
            mask = mask_transform(mask)

        # Process image using the image processor
        if self.image_processor:
            # Convert PIL image to tensor using the image processor
            image = self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

        return image, mask


def prepare_dataloaders(image_folder, mask_folder, hparams, image_processor):
    # Get sorted lists of image and mask files
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder))]
    mask_files = [os.path.join(mask_folder, f) for f in sorted(os.listdir(mask_folder))]

    # Ensure matching number of images and masks
    assert len(image_files) == len(mask_files), "Mismatch between image and mask files."

    # Image augmentations
    augmentations = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    # Create dataset and dataloaders
    dataset = TissueDataset(
        image_files=image_files,
        mask_files=mask_files,
        #augmentations=augmentations,
        image_processor=image_processor,
        border_threshold=hparams["border_threshold"]
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    return train_loader, val_loader, test_loader
