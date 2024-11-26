import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

def verify_data_alignment(dataset, image_files, num_samples=5):
    """
    Verify data processing by displaying original images, preprocessed images, and processed masks.

    Args:
        dataset: Dataset object that provides preprocessed images and masks.
        image_files: List of full paths to the original image files.
        num_samples: Number of samples to visualize.
    """
    for i in range(num_samples):
        # Retrieve the preprocessed image and mask from the dataset
        preprocessed_image, processed_mask = dataset[i]

        # Convert tensors to PIL images for visualization
        preprocessed_image_pil = transforms.ToPILImage()(preprocessed_image)
        processed_mask_pil = transforms.ToPILImage()(processed_mask)

        # Load the original image (image_files already contains the full path)
        original_image = Image.open(image_files[i]).convert("RGB")

        # Display the original image, preprocessed image, and processed mask side by side
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(original_image)
        axs[0].set_title(f"Original Image {i + 1}")
        axs[0].axis("off")

        axs[1].imshow(preprocessed_image_pil)
        axs[1].set_title(f"Preprocessed Image {i + 1}")
        axs[1].axis("off")

        axs[2].imshow(processed_mask_pil, cmap="gray")
        axs[2].set_title(f"Processed Mask {i + 1}")
        axs[2].axis("off")

        plt.show()

import matplotlib.pyplot as plt

def visualize_batch_from_loader(loader, num_batches=1):
    """
    Visualize a few batches of images and masks from the DataLoader.

    Args:
        loader: PyTorch DataLoader providing images and masks.
        num_batches: Number of batches to visualize.
    """
    loader_iter = iter(loader)

    for batch_idx in range(num_batches):
        # Get the next batch of images and masks
        images, masks = next(loader_iter)

        # Move tensors to CPU if necessary and convert to numpy
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        masks_np = masks.squeeze(1).cpu().numpy()  # Remove channel dimension for masks

        # Display images and masks side by side
        batch_size = images_np.shape[0]
        fig, axes = plt.subplots(
            batch_size, 2, figsize=(10, 5 * batch_size)) if batch_size > 1 else plt.subplots( 1, 2, figsize=(10, 5))  # Adjust for batch size of 1

        # Ensure axes is always a 2D array for consistency
        if batch_size == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(batch_size):
            axes[i, 0].imshow(images_np[i])
            axes[i, 0].set_title(f"Image {batch_idx * batch_size + i + 1}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(masks_np[i], cmap="gray")
            axes[i, 1].set_title(f"Mask {batch_idx * batch_size + i + 1}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()
