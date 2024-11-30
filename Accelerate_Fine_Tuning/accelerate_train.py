# Deep learning librairies
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# System librairies
import sys
import os

# Project classes/functions
from accelerate_trainer import train

sys.path.append('..')
from preprocess import create_dataloaders
from model_config import *
from utils import load_hyperparameters, setup_experiment_directories
from loss import ScaledDiceLoss


def main():
    # Initialize accelerator
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    # Get hparam file
    if len(sys.argv) > 1:
        hparam_file = sys.argv[2]
    else:
        raise ValueError("Please provide a path to a hyperparameter file. "
                         "Script usage: python accelerate_train.py --hparams {path/to/hyperparameters.yaml}")

    # Create hyperparameter dict from args and yaml
    hparams = load_hyperparameters(hparam_file)

    # Clear any existing model and cache
    clear_model_and_cache()

    # Set seed
    torch.manual_seed(hparams["seed"])

    # Create model and image processor
    image_processor = load_image_processor(hparams["mask2former_path"])
    model = load_mask2former_model(hparams["mask2former_path"], 2)

    # Freeze required layers and print model parameters
    if accelerator.is_main_process:
        print_trainable_layers(model)

    # Define paths
    image_folder = os.path.join(hparams["data_folder"], "Sliced_Images")
    mask_folder = os.path.join(hparams["data_folder"], "Sliced_masks")

    # Prepare data and get dataloaders
    train_loader, val_loader, _, _, _ = create_dataloaders(image_folder, mask_folder, hparams["batch_size"], image_processor)

    # Directory to save to
    experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "results", hparams["experiment_name"], str(hparams["seed"]))

    # Create experiment folder and setup logging
    if accelerator.is_main_process:
        setup_experiment_directories(experiment_dir, hparam_file)

    # Create optimizer, criterion
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    criterion = ScaledDiceLoss(smooth=hparams["smooth"], threshold_zero_loss=hparams["threshold_zero_loss"])

    # Wrap everything with accelerator
    model, image_processor, optimizer, train_loader, val_loader = accelerator.prepare(
        model, image_processor, optimizer, train_loader, val_loader)

    # Run training
    train(accelerator, model, train_loader, val_loader, criterion,
          optimizer, hparams["epochs"], experiment_dir)
    image_processor.save_pretrained(experiment_dir)


if __name__ == "__main__":
    main()
