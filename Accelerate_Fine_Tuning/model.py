from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import gc
import torch


def clear_model_and_cache():
    """Utility function to delete existing model and optimizer objects and clear GPU memory."""
    if 'model' in globals():
        print("Deleting existing model...")
        del globals()['optimizer']
    gc.collect()
    torch.cuda.empty_cache()


def create_mask2former(num_labels=2):
    # Load the image processor with relevant settings
    image_processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-IN21k-ade-semantic",
        do_rescale=False,
        do_normalize=False,
        do_resize=False
    )

    # Load the Mask2Former model for binary segmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-IN21k-ade-semantic",
        num_labels=num_labels,  # Binary segmentation (background and tissue)
        ignore_mismatched_sizes=True  # Allow resizing of model parameters if dimensions do not match
    )

    return image_processor, model
