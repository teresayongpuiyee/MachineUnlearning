import torch
import numpy as np
import random
from PIL import Image
import argparse
from typing import Tuple, List
import pandas as pd
import os
import logging


def set_seed(
    seed: int
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def device_configuration(
    args: argparse.Namespace
) -> Tuple[torch.device, str]:
    # Device configuration
    if torch.cuda.is_available() and args.gpu:
        device = torch.device("cuda")
        device_name = f"({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device("cpu")
        device_name = ""
    return device, device_name


def image_tensor2image_numpy(
    image_tensor: torch.Tensor,
    squeeze: bool= False,
    detach: bool= False
) -> np.array:
    """
    Input:
        image_tensor= Image in tensor type
        Squeeze = True if the input is in the batch form [1, 1, 64, 64], else False
    Return:
        image numpy
    """
    if squeeze:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
        else:
            #Squeeze from [1, 1, 64, 64] to [1, 64, 64] only if the input is the batch
            image_numpy = image_tensor.cpu().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
    else:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy()  # move tensor to cpu and convert to numpy
        else:
            image_numpy = image_tensor.cpu().numpy() # move tensor to cpu and convert to numpy

    # Transpose the image to (height, width, channels) for visualization
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)

    return image_numpy


def save_tensor(
    image_tensor: torch.Tensor,
    save_path: str
) -> None:
    img_np = image_tensor2image_numpy(image_tensor=image_tensor)
    # Convert to uint8 and scale if necessary
    img_np = (img_np * 255).astype(np.uint8) if img_np.dtype != np.uint8 else img_np
    output_image = Image.fromarray(img_np)
    output_image.save(save_path)


def create_directory_if_not_exists(
    file_path: str,
) -> None:
    # Check the directory exist,
    # If not then create the directory
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory and its parent directories if necessary
        os.makedirs(directory)
        print(f"Created new directory: {file_path}")


def save_model(
    model: torch.nn.Module,
    model_name: str,
    model_root: str,
) -> None:
    create_directory_if_not_exists(f"{model_root}/")
    model_path = f"{model_root}/{model_name}.pt"
    #model_path = f"{model_folder}{model_name}.pt"
    torch.save(model.state_dict(), model_path)


def get_csv_attr(
    csv_path: str,
    column_name: str
) -> List:
    attr_list = []
    df = pd.read_csv(csv_path)
    for attr in df[column_name]:
        attr_list.append(attr)
    return attr_list

def configure_logger(log_file):
    # Define the file where logs will be written
    LOG_FILENAME = log_file

    # Configure the basic settings
    logging.basicConfig(
        filename=LOG_FILENAME,  # Specify the output file
        level=logging.INFO,     # Set the minimum severity level to log (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s' # Define the log message format
        # Example format options:
        # %(asctime)s: Timestamp
        # %(levelname)s: Severity level (e.g., INFO, ERROR)
        # %(name)s: Logger name (default is 'root' if not specified)
        # %(module)s: Module (filename) where the log call was made
        # %(message)s: The actual log message
    )

    logger = logging.getLogger(__name__)

    return logger

def load_pretrained_weights(
    model: torch.nn.Module,
    pretrained_weight: str,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    if len(pretrained_weight):
        logger.info(f"Use pretrained model from {pretrained_weight}")
        state_dict = torch.load(pretrained_weight, map_location=device)

        keys_to_remove = ['fc.weight', 'fc.bias'] 
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key] 
                logger.info(f"✅ Key removed successfully: {key}")
            else:
                logger.info(f"Warning: Expected key '{key}' not found in state_dict.")

        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained model from {pretrained_weight}")
    else:
        logger.info("No pretrained model path provided, training from scratch.")
    
    return model