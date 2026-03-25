from src import utils, dataset, analyse
import argparse
from model import models
from torch.utils.data import DataLoader
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
parser.add_argument("-pretrained_timm", dest="pretrained_timm", action="store_true", default=False, help="Model trained with pretrained timm")
# Model
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
# Unlearn configuration
parser.add_argument("-unlearn_class", type= int, default=0, help= "Class to unlearn")
parser.add_argument("-unlearned_model", type=str, required=True, help="Path to unlearned model")
# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()


def main(args) -> None:
    unlearned_model_path_list = args.unlearned_model.split("/")
    exp_name = unlearned_model_path_list[-3]
    unlearn_method = unlearned_model_path_list[-1].split(".")[0]
    
    model_path = "/".join(unlearned_model_path_list[:-1])
    ori_model_path = f"{model_path}/baseline.pt"
    retrain_model_path = f"{model_path}/retrain.pt"

    output_path = f"./{exp_name}/analyse_outputs/"
    utils.create_directory_if_not_exists(output_path)
    
    logger = utils.configure_logger(f"{output_path}unlearn_{unlearn_method}.log")
    OUTPUT_CONFIG_FILE = f"{output_path}unlearn_{unlearn_method}_config.yaml"
    OUTPUT_METRICS_FILE = f"{output_path}unlearn_{unlearn_method}_metrics.yaml"

    config_dict = vars(args).copy()
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, _ = utils.device_configuration(args=args)

    # Get dataset info e.g., classes and channels
    num_classes, num_channels = dataset.dataset_info(dataset_name= args.dataset)

    unlearned_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)

    # Dataset
    logger.info("Preparing datasets and dataloaders...")
    train_dataset, _ = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root, augment=False, model=unlearned_model, pretrained_timm= args.pretrained_timm
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        dataset=train_dataset,
        unlearn_class=args.unlearn_class
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=False)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False)

    # Model preparation
    logger.info("Loading model checkpoints...")
    ori_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    retrain_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)

    # load checkpoints
    utils.load_model_weights(ori_model, ori_model_path, device)
    utils.load_model_weights(retrain_model, retrain_model_path, device)
    utils.load_model_weights(unlearned_model, args.unlearned_model, device)
    
    # cosine similarity
    logger.info("Computing representation shift alignment metrics...")
    logger.info("On training set...")
    breakdown_train, cos_sim_train, mag_ratio_train, _ = analyse.compute_rep_shift_alignment(
        ori_model, 
        retrain_model, 
        unlearned_model, 
        train_loader, 
        device, 
        unlearn_method, 
        output_path,
        dataset_name="train"
    )

    logger.info("On retain set...")
    breakdown_retain, cos_sim_retain, mag_ratio_retain, mean_reps_retain = analyse.compute_rep_shift_alignment(
        ori_model, 
        retrain_model, 
        unlearned_model, 
        retain_loader, 
        device, 
        unlearn_method, 
        output_path,
        dataset_name="retain"
    )

    logger.info("On forget set...")
    breakdown_unlearn, cos_sim_unlearn, mag_ratio_unlearn, mean_reps_unlearn = analyse.compute_rep_shift_alignment(
        ori_model, 
        retrain_model, 
        unlearned_model, 
        unlearn_loader, 
        device, 
        unlearn_method, 
        output_path,
        dataset_name="unlearn"
    )
    
    logger.info("Calculating harmonic mean of cosine similarities between retain and unlearn sets...")
    cos_sim_h_mean = analyse.calculate_harmonic_mean(cos_sim_retain, cos_sim_unlearn)

    logger.info("Computing forget-retain cosine similarities...")
    unlearn_ret_unl_cos_sim, retrain_ret_unl_cos_sim = analyse.compute_forget_retain_cosine_similarity(mean_reps_retain, mean_reps_unlearn)
    
    dir_align = {
        "train": {"breakdown": breakdown_train, "cosine_similarity": cos_sim_train, "magnitude_ratio": mag_ratio_train},
        "retain": {"breakdown": breakdown_retain, "cosine_similarity": cos_sim_retain, "magnitude_ratio": mag_ratio_retain},
        "unlearn": {"breakdown": breakdown_unlearn, "cosine_similarity": cos_sim_unlearn, "magnitude_ratio": mag_ratio_unlearn},
        "harmonic_mean_retain_unlearn": cos_sim_h_mean,
        "unlearn_ret_unl_cos_sim": unlearn_ret_unl_cos_sim,
        "retrain_ret_unl_cos_sim": retrain_ret_unl_cos_sim
    }
    
    metrics_dict = {
        "Directional Alignment": dir_align,
    }

    logger.info("Saving computed metrics...")
    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)