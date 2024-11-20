# Deep learning librairies
import torch.optim as optim
from accelerate import Accelerator
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# System librairies
import os
import shutil
import argparse
# Project classes/functions
from model import clear_model_and_cache, create_mask2former
from custom_dataset import TissueDataset
from utils import *


def train(accelerator, model, image_processor, train_loader, criterion, optimizer, checkpoint_dir, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        total_iou, total_dice = 0, 0
        for images, masks in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] Training", disable=not accelerator.is_main_process):
            # Process inputs and get model outputs
            image_list = []
            for image in images:
                image_list.append(image)

            inputs = image_processor(images=image_list, return_tensors="pt")
            outputs = model(**inputs)
            tissue_logits = outputs.masks_queries_logits[:,
                            1].requires_grad_()  # Ensure tissue logits require gradients

            target_shape = tissue_logits.shape[-2:]  # Get the height, width from logits shape

            # Resize masks to match output shape
            masks_resized = torch.stack([
                torch.tensor(invert_mask(resize_mask(mask.cpu().numpy(), target_shape)), dtype=torch.float32)
                for mask in masks
            ]).to(accelerator.device)

            # Compute Dice loss
            loss = criterion(tissue_logits, masks_resized)

            # Use accelerator for backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            # Calculate IoU and Dice metrics
            pred_mask = (torch.sigmoid(tissue_logits) > 0.5).float()
            intersection = (pred_mask * masks_resized).sum((1, 2))
            union = pred_mask.sum((1, 2)) + masks_resized.sum((1, 2)) - intersection
            iou = (intersection / (union + 1e-6)).mean().item()
            dice = (2 * intersection / (pred_mask.sum((1, 2)) + masks_resized.sum((1, 2)) + 1e-6)).mean().item()

            total_iou += iou
            total_dice += dice

        avg_loss = running_loss / len(train_loader)
        avg_iou = total_iou / len(train_loader)
        avg_dice = total_dice / len(train_loader)

        # Checkpointing and printing
        if accelerator.is_main_process:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, "
                  f"Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}")

            if len(os.listdir(checkpoint_dir)) > 0:
                previous_checkpoint = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
                shutil.rmtree(previous_checkpoint)

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch+1}"))


def validate(accelerator, model, image_processor, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    total_iou, total_dice = 0, 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation", disable=not accelerator.is_main_process):
            # Convert images to list
            image_list = []
            for image in images:
                image_list.append(image)

            # Process inputs and get model outputs
            inputs = image_processor(images=image_list, return_tensors="pt")
            outputs = model(**inputs)
            tissue_logits = outputs.masks_queries_logits[:, 1]

            target_shape = tissue_logits.shape[-2:]  # Get the height, width from logits shape

            # Resize masks to match output shape
            masks_resized = torch.stack([
                torch.tensor(invert_mask(resize_mask(mask.cpu().numpy(), target_shape)), dtype=torch.float32)
                for mask in masks
            ]).to(accelerator.device)

            # Compute loss
            loss = criterion(tissue_logits, masks_resized)
            val_loss += loss.item()

            # Calculate IoU and Dice metrics
            pred_mask = (torch.sigmoid(tissue_logits) > 0.5).float()
            intersection = (pred_mask * masks_resized).sum((1, 2))
            union = pred_mask.sum((1, 2)) + masks_resized.sum((1, 2)) - intersection
            iou = (intersection / (union + 1e-6)).mean().item()
            dice = (2 * intersection / (pred_mask.sum((1, 2)) + masks_resized.sum((1, 2)) + 1e-6)).mean().item()

            total_iou += iou
            total_dice += dice

    avg_val_loss = val_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_dice = total_dice / len(val_loader)

    if accelerator.is_main_process:
        print(f"Validation Loss: {avg_val_loss:.4f}, Avg IoU: {avg_iou:.4f}, Avg Dice: {avg_dice:.4f}")


def main():
    # Initialize accelerator
    #accelerator = Accelerator(kwargs_handler=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    accelerator = Accelerator()
    # Get arguments
    parser = argparse.ArgumentParser(description="Arguments to fine-tune Mask2Former")
    parser.add_argument("--data_folder", type=str, required=True, help="path/to/data")
    parser.add_argument("--experiment_name", type=str, required=True, help="name of the experiment being run")
    parser.add_argument("--epochs", type=int, required=False, default=10, help="number of epochs to run")
    parser.add_argument("--freeze_classifier", type=bool, required=False, default=False, help="whether to freeze the mask2former classifier")
    parser.add_argument("--freeze_encoder", type=bool, required=False, default=False, help="whether to freeze the mask2former encoder")
    parser.add_argument("--freeze_decoder", type=bool, required=False, default=False, help="whether to freeze the mask2former decoder")

    args = parser.parse_args()

    # Clear any existing model and cache
    clear_model_and_cache()

    # Create model and image processor
    image_processor, model = create_mask2former()

    # Freeze some layers in the model if wanted
    for name, param in model.named_parameters():
        if args.freeze_classifier:
            if "class_predictor" in name:
                param.requires_grad = False
        elif args.freeze_encoder:
            if "class_encoder" in name:
                param.requires_grad = False
        elif args.freeze_decoder:
            if "class_decoder" in name:
                param.requires_grad = False
        else:
            param.requires_grad = True

    # Display the trainable layers for confirmation
    if accelerator.is_main_process:
        print("Trainable layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name} is trainable")
            else:
                print(f"\n{name} is not trainable\n")

    # Define paths
    image_folder = os.path.join(args.data_folder, "Sliced_Images")
    mask_folder = os.path.join(args.data_folder, "Sliced_masks")

    # Get sorted lists of image and mask files
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # Ensure matching number of images and masks
    assert len(image_files) == len(mask_files), "Mismatch between image and mask files."

    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Mask transform - avoid normalization if not needed
    mask_transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloaders
    dataset = TissueDataset(
        image_files, mask_files,
        image_folder, mask_folder,
        transform=image_transform,
        mask_transform=mask_transform
    )

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create some directories to save to
    experiment_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", args.experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Create criterion, optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = DiceLoss()

    # Pass everything through accelerator
    model, image_processor, optimizer, train_loader, val_loader = accelerator.prepare(
        model, image_processor, optimizer, train_loader, val_loader
    )

    # Run training and validation
    train(accelerator, model, image_processor, train_loader, criterion, optimizer, checkpoint_dir, args.epochs)
    validate(accelerator, model, image_processor, val_loader, criterion)


if __name__ == "__main__":
    main()
