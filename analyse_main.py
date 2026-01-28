from src import utils, dataset
import argparse
from model import models
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import yaml

parser = argparse.ArgumentParser()
# Device
parser.add_argument("-gpu", type= bool, default= True, help= "use gpu or not")
# Dataset
parser.add_argument("-root", type= str, default= "./data", help= "Dataset root directory")
parser.add_argument("-dataset", type= str, help= "Dataset configuration",
                    choices=["MNist",
                             "FMNist",
                             "Cifar10",
                             "Cifar100",
                             "TinyImagenet"])
# Model
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")

# Unlearn configuration
parser.add_argument("-unlearn_class", type= int, default=0, help= "Class to unlearn")

parser.add_argument("-model_a", type=str, required=True, help="Path to model A (original)")
parser.add_argument("-model_b", type=str, required=True, help="Path to model B (e.g. retrained)")
parser.add_argument("-model_c", type=str, required=True, help="Path to model C (e.g. unlearned)")

# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")

# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()

@torch.no_grad()
def extract_mean_representation(model, dataloader, device):
    model.eval()
    reps = []

    for x, _ in dataloader:
        x = x.to(device)
        h = model.feature_extractor(x)   
        h = h.view(h.size(0), -1)  # (N, D)
        reps.append(h)

    reps = torch.cat(reps, dim=0)        # (total_N, D)
    mean_rep = reps.mean(dim=0)           # (D,)
    return mean_rep

def calculate_cos_sim(model_a, model_b, model_c, dataloader, device):
    # extract mean representations - train loader
    mean_a = extract_mean_representation(model_a, dataloader, device)
    mean_b = extract_mean_representation(model_b, dataloader, device)
    mean_c = extract_mean_representation(model_c, dataloader, device)

    # compute representation differences
    diff_ab = mean_a - mean_b
    diff_ac = mean_a - mean_c

    # cosine similarity
    cos_sim = F.cosine_similarity(diff_ab.unsqueeze(0), diff_ac.unsqueeze(0)).item()

    return cos_sim

def main(args) -> None:
    exp_name = args.model_a.split("/")[-3]
    config_dict = vars(args).copy()

    logger = utils.configure_logger(f"./{exp_name}/unlearn_dir_align.log")

    OUTPUT_CONFIG_FILE = f"./{exp_name}/unlearn_dir_align_config.yaml"
    OUTPUT_METRICS_FILE = f"./{exp_name}/unlearn_dir_align_metrics.yaml"
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, device_name = utils.device_configuration(args=args)

    # Dataset
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=train_dataset,
        unlearn_class=args.unlearn_class
    )

    test_retain_dataset, test_unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=test_dataset,
        unlearn_class=args.unlearn_class
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=args.batch_size, shuffle=True)
    test_unlearn_loader = DataLoader(test_unlearn_dataset, batch_size=args.batch_size, shuffle=True)

    # Model preparation
    model_a = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    model_b = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    model_c = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)

    # load checkpoints
    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))
    model_c.load_state_dict(torch.load(args.model_c, map_location=device))

    cos_sim_train = calculate_cos_sim(model_a, model_b, model_c, train_loader, device)
    cos_sim_retain = calculate_cos_sim(model_a, model_b, model_c, retain_loader, device)
    cos_sim_unlearn = calculate_cos_sim(model_a, model_b, model_c, unlearn_loader, device)

    metrics_dict = {
        "Cosine similarity train samples shift": cos_sim_train,
        "Cosine similarity retain samples shift": cos_sim_retain,
        "Cosine similarity forget samples shift": cos_sim_unlearn,
    }

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)