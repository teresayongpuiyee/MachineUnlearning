import torch
from typing import Tuple, List
from torch.utils.data import ConcatDataset, random_split
from tqdm import tqdm
from src import raw_dataset
from torch.utils.data import Dataset, Subset


def get_dataset(
    dataset_name: str,
    root: str,
    augment: bool= True,
):
    train_dataset = getattr(raw_dataset, dataset_name)(
        root= root, train= True, download= True, augment= augment
    )
    test_dataset = getattr(raw_dataset, dataset_name)(
        root=root, train= False, download= True
    )
    # Get dataset info e.g., classes and channels
    num_classes, num_channels = dataset_info(dataset_name= dataset_name)
    return train_dataset, test_dataset, num_classes, num_channels


def dataset_info(
    dataset_name: str
)-> Tuple[int, int]:

    dataset_dict = {
        "MNist": {"num_classes": 10, "num_channels": 1},
        "FMNist": {"num_classes": 10, "num_channels": 1},
        "Cifar10": {"num_classes": 10, "num_channels": 3},
        "Cifar100": {"num_classes": 100, "num_channels": 3},
        "TinyImagenet":{"num_classes": 200, "num_channels": 3}
    }
    num_classes = dataset_dict[dataset_name]["num_classes"]
    num_channels = dataset_dict[dataset_name]["num_channels"]

    return num_classes, num_channels

def split_unlearn_dataset(dataset, unlearn_class):
    # 1. Convert targets to a tensor for fast filtering
    # Works for CIFAR and most torchvision datasets
    targets = torch.tensor(dataset.targets)
    
    # 2. Find indices
    retain_indices = (targets != unlearn_class).nonzero(as_tuple=True)[0].tolist()
    unlearn_indices = (targets == unlearn_class).nonzero(as_tuple=True)[0].tolist()
    
    # 3. Create Subsets (this does NOT copy the images, only the indices)
    retain_ds = Subset(dataset, retain_indices)
    unlearn_ds = Subset(dataset, unlearn_indices)
    
    return retain_ds, unlearn_ds


def inject_square(
    input_tensor: torch.Tensor,
    init_coor: int,
    square_size: int,
    color: str
)-> torch.Tensor:

    if color not in ["red", "green", "blue"]:
        raise Exception("Choose correct color")

    if color == "red":
        color_list = [0.5, 0, 0]
    elif color == "green":
        color_list = [0, 0.5, 0]
    else:
        color_list = [0, 0, 0.5]

    input_tensor[0, init_coor:square_size + init_coor, init_coor:square_size + init_coor] = color_list[0]  # Red channel
    input_tensor[1, init_coor:square_size + init_coor, init_coor:square_size + init_coor] = color_list[1]  # Green channel
    input_tensor[2, init_coor:square_size + init_coor, init_coor:square_size + init_coor] = color_list[2]  # Blue channel

    return input_tensor


class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y