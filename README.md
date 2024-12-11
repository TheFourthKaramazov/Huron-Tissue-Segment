# Tissue Segmentation using Mask2Former

This project demonstrates a tissue segmentation pipeline built using the **Mask2Former model** from Facebook AI. It performs binary segmentation on medical imaging data from Huron Pathalogy, distinguishing between the background and tissue regions.

## Metrics Acheived

- Mean IoU: 0.8032
- Mean Dice: 0.8721
- Mean Pixel Accuracy: 0.9743
  
# For Graders:

## Final Trained model download link: [https://drive.google.com/drive/folders/1-041y6yOD-oFiB2gl0ltAiOwN2Ul0drQ?usp=sharing](https://drive.google.com/drive/folders/1-041y6yOD-oFiB2gl0ltAiOwN2Ul0drQ?usp=sharing)

## SAMPLE DATASET INCLUDED IN REPOSITORY: SIMPLY CLONE ENTIRE REPOSITORY 

## SAMPLE DATSET FUNCTIONS WITH `example_inference_SAMPLE_DATASET.ipynb`. ALL OTHER SCRIPTS REQUIRE FULL DATASET FOUND IN data/link folder. ENSURE MODEL IS DOWNLOADED FROM GOOGLE LINK IN CURRENT WORKING DIRECTORY UNDER models/

### Note: notebooks are already executed. Results can be observed without running them again if desired.

### 1. Clone entire repository and pip install dependencies outlined below. (example notebooks rely on python files)

### 2. Refer to *usage_example.ipynb* for full model training. Execute each cell one after another. (dataset is divided by 50 so training will be fast)

### 3. Refer to *example_usage_google_colab.ipynb* for training on full dataset with grid search parameters which resulted in best performing model. (8 hours training time on L4 GPU)

### 4. Refer to *example_inference_only.ipynb* for inferencing only.  Execute each cell one after another. Model is the result of step 3. above and must be downloaded from google link (too large for github). Ensure path and name correpond to loading script.

### 5. Both scripts plot final results for comparison of original image, predicted mask, ground-truth mask for evaluation.(Files must be opened to see)

### 6. All notebooks rely on data being downloaded and set according to the dataset structure outlined below (google colab notebook for full model training relys mounted drive, refer to corresponding file)



## Features

- Pretrained **Mask2Former model** fine-tuned for binary segmentation.
- Support for custom datasets with images and masks.
- Efficient data preprocessing, training, and inference pipelines.
- Visualization of results for training, validation, and testing.

## Setup Instructions

### Install Dependencies

Ensure you have the required libraries installed. Use the following dependencies: `matplotlib`, `torch`, `numpy`, `tqdm`, `Pillow`, `torchvision`, `transformers`, `gc`, `os`, `scipy`. You can install these dependencies using pip:

```
bash
pip install matplotlib torch numpy tqdm Pillow torchvision transformers scipy 
```

### Dataset Structure

The dataset is organized as follows:
```
data/
├── Huron_data/
│   ├── Sliced_Images/      # Image folder
│   ├── Sliced_Masks/       # Corresponding binary masks
```

### Pretrained Model

The project uses the **Mask2Former pretrained model** (`facebook/mask2former-swin-base-IN21k-ade-semantic`). The model is configured for binary segmentation (two classes: tissue and background).

## Usage

### Refer to usage_example.ipynb

#### All details are laid out including optimal paramters and using the corresponding python files. Comments outline entire approach.

## Example Workflow

1. Preprocess Data: Ensure images and masks are aligned and formatted. Use the `verify_data_alignment()` function to validate consistency.
2. Train the Model: Use the `train()` function to fine-tune the Mask2Former model on your dataset.
3. Plot metrics over epochs (loss, mIoU, dice, pixel accuracy) with `visualize_training_metrics()` and `visualize_validation_metrics()` from `data_visualization.py`.
4. Save/Load the Model: Save the model and processor using the provided utility functions.
5. Run Inference: Perform predictions and visualize results using `infer_and_display()`.

#### Note: Please refer to usage_example.ipynb for full details.

## Performance and Metrics

- **Metrics**: Pixel accuracy, mIoU, dice coefficient.
- **Loss Function**: Best performance with **Scaled Dice Loss custom class** (Details in loss.py)

##### Note: for this task, the qualitative metrics are more important than the quantiative metrics. Use infer_and_display() to inspect your results. You will notice a nearly perfect match with predictions and ground truths.


### Hyper-parameter Tuning

The hyperparameter tuning script is uses `optuna`, a popular optimization library, to fine tune a few hyperparameters. This is done by creating a *study* which runs multiple trials sequentially. In each trial, the `objective` function
is ran and returns the mIoU which is what is then used to assess the performance of a trial. This can take a while to run depending on the number of trials, epochs, and the size of the train subset used.

To install optuna, you can run 
```
pip install optuna
```

Finally, simply make sure you have dataset downloaded into the root folder using the dataset structure shown above.

### Accelerated Training 

The files in Accelerate_Fine_Tuning are built to train the Mask2Former model on Hugging Face's [Accelerate library](https://huggingface.co/docs/accelerate/v0.3.0/index.html). This allows users to run multi-GPU training easily. Below are the steps to set up and execute the script. The following steps assume you have already followed the steps for general training, including installing the dependencies and downloading the dataset. 

#### Additional Dependencies

For this code you will only need to install `Accelerate` and `pyyaml`. You can install them as follows:
```
pip install accelerate pyyaml
```

#### Accelerator setup

To use Accelerate, you typically need to run `accelerate config` and select from a series of options. You may do this if you prefer but in the Accelerate directory, there is a default_config.yaml that can be used (this configuration is tested and is working, if you use a custom configuration, it may break). The only field you may need to change in default_config.yaml is the `num_processes` which is the number of available GPUs. 

#### Hyperparameters

For this script, you will need to create or adapt the hyperparameters.yaml file. Each hyperparameter can be tweaked to your liking. Hyperparameters can also be overwritten from the command line using `--hyperparameter_name value`. 

#### Running the script

Now that everything is setup, you can run the training script. Use the following command if you are using the default_config.yaml: 

```
accelerate launch --config_file default_config.yaml accelerate_train.py --hparams hyperparameters.yaml --data_folder path/to/data --other args
```

The training log and model checkpoints will be saved as follows:
```
results/
├── seed/
│   ├── training.log         # Training log with loss, dice, IOU and pixel accuracy per epoch
│   ├── best_iou_checkpoint/ # Folder with saved model checkpoint
│   ├── hyperparameters.yaml # Hyperparameters used for this experiment
```

## Requirements

- Python >= 3.8
- GPU recommended for large datasets.
- On CPU, FUll dataset training will take 2 hours per epoch, while on GPU
- On GPU, Full dataset training will take ~5 to 30 minutes per epoch depending on the GPU
- A100 takes 5 minutes per epoch, while T4 takes 30 minutes per epoch
- Optimal performance requires at least 10 epochs.

### Full Dependency List

##### This project uses the following imports:

###### Note: although these imports are required, they are taken care of in the python files and usage_example.ipynb file. Only the libraries must be installed on your system.

```python
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import gc
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
```
## References

1. [Mask2Former on Hugging Face](https://huggingface.co/facebook/mask2former)

## Troubleshooting

- Ensure dataset paths are correct.
- For large datasets, use a GPU for faster training and inference.
- If IoU/Dice scores are low, check that images and masks are aligned properly.

## Acknowledgments

This project leverages the **Mask2Former model** from Facebook AI Research and is implemented using PyTorch and Hugging Face Transformers.

The Mask2Former github page can be found here

[Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation](https://github.com/facebookresearch/Mask2Former)
