import torch
import gc
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

def clear_model_and_cache():
    """
    Utility function to delete existing model and optimizer objects
    and clear GPU memory to avoid memory leaks.
    """
    if 'model' in globals():
        print("Deleting existing model...")
        del globals()['model']
        del globals()['optimizer']
    gc.collect()
    torch.cuda.empty_cache()

def load_image_processor(pretrained_model_name, do_rescale=True, do_normalize=False, do_resize=True):
    """
    Load the Mask2Former image processor with specified settings.

    Parameters:
    - pretrained_model_name: Hugging Face model name.
    - do_rescale: Whether to rescale image values.
    - do_normalize: Whether to normalize image values.
    - do_resize: Whether to resize the images.

    Returns:
    - image_processor: The configured image processor object.
    """
    image_processor = Mask2FormerImageProcessor.from_pretrained(
        pretrained_model_name,
        do_rescale=do_rescale,
        do_normalize=do_normalize,
        do_resize=do_resize
    )
    return image_processor

def load_mask2former_model(pretrained_model_name, num_labels, ignore_mismatched_sizes=True, freeze_encoder=True):
    """
    Load the Mask2Former model for universal segmentation.

    Parameters:
    - pretrained_model_name: Hugging Face model name.
    - num_labels: Number of segmentation labels (e.g., 2 for binary segmentation).
    - ignore_mismatched_sizes: Whether to allow resizing of model parameters.
    - freeze_encoder: Whether to freeze the encoder backbone.

    Returns:
    - model: The Mask2Former model object.
    """
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )

    if freeze_encoder:
        # Freeze the encoder backbone
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    return model

def print_trainable_layers(model):
    """
    Display which layers of the model are trainable or frozen.

    Parameters:
    - model: The Mask2Former model.
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} is trainable")
        else:
            print(f"{name} is frozen")