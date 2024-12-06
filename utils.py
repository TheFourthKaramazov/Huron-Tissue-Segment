import os
import logging
import shutil
import argparse
import yaml
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


def setup_experiment_directories(experiment_dir, hparam_file):
    logs_path = os.path.join(experiment_dir, "training.log")

    # Create experiment folder
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Results and checkpoints will be saved to {experiment_dir}")

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler(logs_path)])

    # Copy hparams to project dir for future reference
    shutil.copy(hparam_file, experiment_dir)


def load_hyperparameters(yaml_file):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Arguments to fine-tune Mask2Former")
    parser.add_argument("--hparams", type=str, required=True)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--data_folder", type=str, required=False)
    parser.add_argument("--experiment_name", type=str, required=False)
    parser.add_argument("--lr", type=float, required=False)
    parser.add_argument("--weight_decay", type=float, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--epochs", type=int, required=False)
    parser.add_argument("--smooth", type=float, required=False)
    parser.add_argument("--threshold_zero_loss", type=int, required=False)

    args = parser.parse_args()

    # Load hyperparameters from YAML file
    with open(yaml_file, 'r') as file:
        hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

    # Override hyperparameters from command-line arguments
    for key, value in hyperparameters.items():
        if key in args.__dict__ and args.__dict__[key] is not None:
            hyperparameters[key] = args.__dict__[key]

    return hyperparameters
