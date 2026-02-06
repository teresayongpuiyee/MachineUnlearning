from src import utils
import argparse
from src import dataset, metrics, repr_metrics
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
# Model
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
parser.add_argument("-unlearned_model", type=str, required=True, help="Path to unlearned model")
# Unlearn configuration
parser.add_argument("-unlearn_class", type= int, help= "Class to unlearn")
# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()


def main(args) -> None:
    unlearned_model_path_list = args.unlearned_model.split("/")
    exp_name = unlearned_model_path_list[-3]
    unlearn_method = unlearned_model_path_list[-1].split(".")[0]

    output_path = f"./{exp_name}/mia_evaluate/"
    utils.create_directory_if_not_exists(output_path)
    
    logger = utils.configure_logger(f"{output_path}unlearn_{unlearn_method}.log")
    OUTPUT_CONFIG_FILE = f"{output_path}unlearn_{unlearn_method}_config.yaml"
    OUTPUT_METRICS_FILE = f"{output_path}unlearn_{unlearn_method}_metrics.yaml"
    
    config_dict = vars(args).copy()
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Evaluating unlearning method {unlearn_method} on dataset {args.dataset}")
    
    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, _ = utils.device_configuration(args=args)
    
    # Dataset
    logger.info("Preparing datasets and dataloaders...")
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root, augment=False
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        dataset=train_dataset,
        unlearn_class=args.unlearn_class
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=False)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model preparation
    logger.info("Loading model checkpoints...")
    unlearned_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    utils.load_model_weights(model=unlearned_model, model_path=args.unlearned_model,device=device)

    # Evaluation after unlearning
    # Classification-level evaluation
    logger.info(f"Logit MIA evaluation...")
    # Bad Teacher MIA
    badt_mia = metrics.badt_mia(
        retain_loader=retain_loader,
        forget_loader=unlearn_loader,
        test_loader=test_loader,
        model=unlearned_model)
    logger.info(f"Bad T MIA: {badt_mia}")

    # Representation-level evaluation
    train_reps, train_labels = repr_metrics.get_representations(train_loader, unlearned_model)
    test_reps, test_labels = repr_metrics.get_representations(test_loader, unlearned_model)
    retain_reps, retain_labels = repr_metrics.get_representations(retain_loader, unlearned_model)
    forget_reps, _ = repr_metrics.get_representations(unlearn_loader, unlearned_model)

    logger.info(f"Representation MIA evaluation...")
    # Bad Teacher equivalent Rep-MIA with balance and normalize features
    badt_rep_mia_metrics, badt_rep_mia_asr = repr_metrics.badt_rep_mia(
        retain_reps=retain_reps,
        forget_reps=forget_reps,
        test_reps=test_reps,
        retain_labels=retain_labels
    )
    logger.info(f"Bad T rep-MIA: {badt_rep_mia_asr}")

    # POUR
    pour_rmia_metrics, pour_rmia_asr = repr_metrics.pour_rmia(
        train_reps=train_reps,
        test_reps=test_reps,
        train_labels=train_labels,
        test_labels=test_labels,
        unlearn_class=args.unlearn_class,
    )
    logger.info(f"POUR rMIA: {pour_rmia_asr}")

    # SURE
    sure_miars_metrics, sure_miars_asr = repr_metrics.sure_miars(
        train_reps=train_reps,
        test_reps=test_reps,
        train_labels=train_labels,
        test_labels=test_labels,
        unlearn_class=args.unlearn_class,
    )
    logger.info(f"SURE MIARS: {sure_miars_asr}")
    
    repr_metrics.visualize_tsne(
        reps=train_reps,
        all_labels=train_labels,
        unlearn_method=unlearn_method,
        exp_name=exp_name
    )
    logger.info("t-SNE visualization saved.")

    metrics_dict = {
        "classification/badt_mia": float(badt_mia),
        
        # attack model metrics
        "representation/badt_rep_mia": badt_rep_mia_metrics,
        "representation/pour_rmia": pour_rmia_metrics,
        "representation/sure_miars": sure_miars_metrics,
        
        # forget asr
        "representation/badt_rep_mia_asr": badt_rep_mia_asr,
        "representation/pour_rmia_asr": pour_rmia_asr,
        "representation/sure_miars_asr": sure_miars_asr,
    }

    logger.info("Saving computed metrics...")
    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)