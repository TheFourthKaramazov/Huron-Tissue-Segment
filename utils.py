import os
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

def save_model_and_processor(model, image_processor, save_dir):
    """
    Save the trained model and image processor to a specified directory.

    Args:
        model: The trained Mask2Former model.
        image_processor: The corresponding image processor used for preprocessing.
        save_dir: Path to the directory where the model and processor will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)
    print(f"Model and image processor saved to {save_dir}")

def load_model_and_processor(save_dir):
    """
    Load a trained Mask2Former model and its corresponding image processor from a directory.

    Args:
        save_dir: Path to the directory where the model and processor are stored.

    Returns:
        model: The loaded Mask2Former model.
        image_processor: The loaded image processor.
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Directory {save_dir} does not exist.")

    model = Mask2FormerForUniversalSegmentation.from_pretrained(save_dir)
    image_processor = Mask2FormerImageProcessor.from_pretrained(save_dir)
    print(f"Model and image processor loaded from {save_dir}")
    return model, image_processor