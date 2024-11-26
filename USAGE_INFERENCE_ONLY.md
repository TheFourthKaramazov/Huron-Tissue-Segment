# USAGE_INFERENCE.md

## Usage Guide for Running Inference

This guide explains how to set up your environment, prepare data loaders, load a pre-trained model, and perform inference on a validation/test set using the provided scripts.

### Import necessary libraries

```
python

import os

from model_configimport *

from preprocessimport *

from train import *

from utils import *

from inference import *

from torch.optim import AdamW

from loss import *

from data_visualization import *
```

### Load Data

```
python
# Define paths for the dataset
image_folder = "data/Huron_data/Sliced_Images"
mask_folder = "data/Huron_data/Sliced_Masks"

# Ensure the dataset folders exist
assert os.path.exists(image_folder), "Image folder not found!"
assert os.path.exists(mask_folder), "Mask folder not found!"
```

### 2. Create DataLoaders for Training and Validation

Use the `create_dataloaders` function to create training and validation data loaders. This prepares the data for model evaluation.

```train_loader,
python
train_loader, val_loader, dataset, image_files, mask_files = create_dataloaders(
    image_folder=image_folder,
    mask_folder=mask_folder,
    image_processor=image_processor,
    batch_size=16,
    divide=50
)
```

#### Note. The size of divide parameter will determine the length of the valdidation set you are inferencing on.

#### Note. The built in functions for preprocessing the data rely on there being a training dataset. If you attempt to create the validation set without a training set, no preprocessing will be done.

# Verify the DataLoaders

```
python

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of training examples: {len(train_loader.dataset)}")
print(f"Number of validation examples: {len(val_loader.dataset)}")

```

---

### 2. Load the Pre-trained Model

Specify the path to the saved model and load it using the `load_model_and_processor` function.

```model_path
python
model_test, image_processor_test = load_model_and_processor(model_path)
```

---

### 3. Set the Device

Check for GPU availability and set the device for inference.

```device
python
print(f"Device: {device}")
```

---

### 4. Perform Inference

Use the `infer_and_display` function to run inference on the test set and visualize predictions.

```
python
infer_and_display(
    model=model_test,
    image_processor=image_processor_test,
    dataloader=val_loader,
    device=device,
    num_samples=30,
    target_size=(256, 256)
)
```

---

### Notes

- Ensure your dataset paths (`image_folder` and `mask_folder`) and the saved model path (`model_path`) are correctly configured before running the script.
- Adjust the `divide` parameter to control dataset size for faster testing during development.
- GPU is recommended for faster inference.
