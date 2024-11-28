# Deep learning librairies
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
# System librairies
import sys
import logging
# Project classes/functions
from model import clear_model_and_cache, create_mask2former
from utils import *
from trainer import train, test
from custom_dataset import prepare_dataloaders

def main():
    # Initialize accelerator
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    # Get hparam file
    if len(sys.argv) > 1:
        hparam_file = sys.argv[2]
    else:
        raise ValueError("Please provide a path to a hyperparameter file. "
                         "Script usage: python train.py --hparams {path/to/hyperparameters.yaml}")

    # Create hyperparameter dict from args and yaml
    hparams = load_hyperparameters(hparam_file)

    # Clear any existing model and cache
    clear_model_and_cache()

    # Set seed
    torch.manual_seed(hparams["seed"])

    # Create model and image processor
    image_processor, model = create_mask2former(hparams["num_labels"])

    # Freeze required layers and print model parameters
    freeze_and_prepare(model, accelerator, hparams)

    # Define paths
    image_folder = os.path.join(hparams["data_folder"], "Sliced_Images")
    mask_folder = os.path.join(hparams["data_folder"], "Sliced_masks")

    # Prepare data and get dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(image_folder, mask_folder, hparams, image_processor)

    # Directory to save to
    experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "results", hparams["experiment_name"], str(hparams["seed"]))

    # Create experiment folder and setup logging
    setup_experiment_directories(experiment_dir, accelerator, hparam_file)

    # Create criterion, optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hparams["lr"])

    if hparams["criterion"] == "dice":
        criterion = DiceLoss()
    elif hparams["criterion"] == "bce_dice":
        criterion = CombinedDiceBCELoss()
    else:
        raise ValueError(f"{hparams['criterion']} is not a valid criterion function")

    # Print trainable parameters
    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            print(f"{name}, requires grad: {param.requires_grad}")

    # Pass everything through accelerator
    model, image_processor, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, image_processor, optimizer, train_loader, val_loader, test_loader
    )

    # Run training
    train(accelerator, model,
          train_loader, val_loader, criterion,
          optimizer, hparams["epochs"], experiment_dir)
    image_processor.save_pretrained(experiment_dir)

    # Final test
    test(accelerator, model, test_loader, criterion)


if __name__ == "__main__":
    main()
