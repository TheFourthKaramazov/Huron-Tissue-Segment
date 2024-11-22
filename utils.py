import os
import PIL
import cv2
import torch
import numpy as np
from PIL import ImageEnhance, Image
from conch.open_clip_custom import create_model_from_pretrained

# Global variables for the conch model
conch_model = None
conch_preprocess = None


def convert_to_numpy(image):
    """
    Convert either an image path, a PIL image, a torch tensor or a numpy array to a numpy array.

    :param image: The image to convert
    :return: The numpy array
    """
    if isinstance(image, str):
        return np.array(Image.open(image))
    elif isinstance(image, Image.Image):
        return np.array(image)
    elif isinstance(image, torch.Tensor):
        return image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise ValueError("Invalid image type. Must be either a path, PIL image, torch tensor or numpy array.")


def dice_score(gray_mask, gray_image):
    """
    Compute the DICE score between the gray mask and the gray image.

    :param gray_mask: ground truth gray scale mask
    :param gray_image: gray scale image
    :return: DICE score
    """
    mask_gt_0 = gray_mask > 0
    gray_image_gt_0 = gray_image > 0
    intersection = np.sum(mask_gt_0 & gray_image_gt_0)
    combined = np.sum(mask_gt_0) + np.sum(gray_image_gt_0)

    if combined == 0:  # no tissue whatsoever
        return 1

    # TODO: what do to do if combined is 1 and intersection is 0? is it really 0% quality???
    # TODO: should we add some kind of smoothing to the denominator/nominator?
    return 2 * intersection / combined


def load_model(hf_token=None):
    global conch_model, conch_preprocess

    if os.environ.get("HF_TOKEN") is None and hf_token is None:
        raise ValueError("You need to provide a Hugging Face API token to load the model.")

    conch_model, conch_preprocess = create_model_from_pretrained(
        'conch_ViT-B-16',
        "hf_hub:MahmoodLab/conch",
        hf_auth_token=hf_token
    )


def image_to_embedding(image):
    # ensure that the conch model is loaded
    if conch_model is None or conch_preprocess is None:
        raise ValueError("You need to load the model first (call load_model(<your hugging face token>)).")

    image = convert_to_numpy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = conch_preprocess(image).unsqueeze(0)
    with torch.inference_mode():
        image_embs = conch_model.encode_image(image, proj_contrast=False, normalize=False)
    return image_embs


def get_mask_quality(image, mask) -> float:
    """
    Estimates the quality of a mask by comparing it to the original image using
    techniques shown in data_visualization.ipynb. Simply, this is done by comparing how much the colored
    pixels in the image correspond to the positive pixels in the mask. The more the colored pixels correspond
    to the positive pixels in the mask, the better the mask is.

    Note: If the mask and image is completely empty, the quality is set to 1 as the mask is perfect.

    :param image: The original image
    :param mask: The mask to be evaluated
    :return: The quality of the mask
    """
    image = convert_to_numpy(image)
    mask = convert_to_numpy(mask)

    # Apply blur to make close dark random colors blend into colors that are closer to the grayscale
    image = cv2.blur(image, (3, 3))

    # **MAGIC**
    # Grayscale: when the RGB/BGR values are close to each other (from white to grey to black)
    # We set a pixel value to 1 if its RGB/BGR values are far from the grayscale otherwise 0
    image_H, image_W, image_C = image.shape
    gray_mask = (np.abs(image - image.mean(axis=2).reshape(image_H, image_W, 1).repeat(image_C, axis=2)) >= 10).any(
        axis=2)

    # Get the dice score between the gray mask and the gray image and return it as the quality
    return dice_score(gray_mask, mask)


def crop_black_borders(image: Image, threshold=30):
    """
    Crop black borders from an image iteratively based on a threshold for black pixels.

    :param image: The image to crop
    :param threshold: The threshold for black pixels
    :return: The cropped image
    """
    img_array = np.array(image)
    gray_img = np.mean(img_array, axis=2)  # Convert to grayscale by averaging channels

    # Initialize cropping boundaries
    top, bottom = 0, gray_img.shape[0]
    left, right = 0, gray_img.shape[1]

    # Crop from the top
    while top < bottom and np.mean(gray_img[top, :]) <= threshold:
        top += 1

    # Crop from the bottom
    while bottom > top and np.mean(gray_img[bottom - 1, :]) <= threshold:
        bottom -= 1

    # Crop from the left
    while left < right and np.mean(gray_img[:, left]) <= threshold:
        left += 1

    # Crop from the right
    while right > left and np.mean(gray_img[:, right - 1]) <= threshold:
        right -= 1

    # Crop the image to the calculated bounds
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def _preprocess_image_new(image) -> np.ndarray:
    """
    Preprocesses an image to be passed through mask2former using the new technique.

    :param image: The image to be preprocessed
    :return: The preprocessed image
    """
    raise NotImplementedError("The new technique is not implemented yet.")
    #image = convert_to_numpy(image)
    #return image


def _preprocess_image_current(image) -> np.ndarray:
    """
    Preprocesses an image to be passed through mask2former using the current technique.

    :param image: The image to be preprocessed
    :return: The preprocessed image
    """
    image = convert_to_numpy(image)
    image = Image.fromarray(image).convert("RGB")
    cropped_image = crop_black_borders(image)

    # Significantly enhance contrast
    enhancer = ImageEnhance.Contrast(cropped_image)
    enhanced_image = enhancer.enhance(10).convert("BGR")

    return np.array(enhanced_image)


def preprocess_mask(mask, to_binary=True) -> np.ndarray:
    """
    Preprocesses a mask to be passed through mask2former.

    :param mask: The mask to be preprocessed (shape: (H, W, C))
    :param to_binary: Whether to convert the mask to binary or to 0, 255
    :return: The preprocessed mask
    """
    mask = convert_to_numpy(mask)
    gray_mask = np.mean(mask, axis=2)  # Convert to grayscale by averaging channels
    if to_binary:
        return (gray_mask > 0).astype(np.float32)
    else:
        return ((gray_mask > 0) * 255).astype(np.float32)


def preprocess_image(image, use_new_technique: bool = False) -> np.ndarray:
    """
    Preprocesses a image to be passed through mask2former.
    If use_new_technique is True, the new technique will be used.
    This new technique removes black borders from the image by replacing them with white pixels
    whereas the old technique crops the image which generates a smaller image with arbitrary dimensions.

    :param image: The image to be preprocessed
    :param use_new_technique: Whether to use the new technique or not
    :return: The preprocessed image
    """
    if use_new_technique:
        return _preprocess_image_new(image)
    else:
        return _preprocess_image_current(image)
