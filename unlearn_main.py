from src import utils
import argparse
from src import dataset, metrics, repr_metrics
from model import models
from torch.utils.data import DataLoader
import torch
from unlearn_strategies import strategies
import time
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
parser.add_argument("-model_root", type= str, default= "checkpoint", help= "Dataset root directory")
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
parser.add_argument("-save_model", dest="save_model", action="store_true", default= False, help= "Save trained model option")
parser.add_argument("-retrain_pretrained_weight", type= str, default= "", help= "Pretrained model path")

# Unlearn configuration
parser.add_argument("-unlearn_method", type= str, default= "lipschitz",
                    choices= ["baseline",
                              "retrain",
                              "fine_tune",
                              "gradient_ascent",
                              "bad_teacher",
                              "scrub",
                              "amnesiac",
                              "boundary",
                              "ntk",
                              "fisher",
                              "unsir",
                              "ssd"],
                    help= "Baselines unlearn method")

parser.add_argument("-model_path", type= str,
                    help= "Trained model path")
parser.add_argument("-unlearn_class", type= int, help= "Class to unlearn")

# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-scenario", type= str, default= "class",
                    choices= ["class", "client", "sample"], help= "Training and unlearning scenario")

# Unlearn Hyperparameter
parser.add_argument("-linear_probe_lr", type=float, default= 1e-4, help='Learning rate')

# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

parser.add_argument("-tsne", dest="tsne", action="store_true", default=False, help="Enable t-SNE visualization")

args = parser.parse_args()


def main(args) -> None:
    exp_name = args.model_path.split("/")[-3]
    config_dict = vars(args).copy()

    # Read train config for retrain method
    if args.unlearn_method == "retrain":
        with open(f"./{exp_name}/train_config.yaml", 'r') as f:
            train_config = yaml.safe_load(f)

        config_dict.update(train_config)
        # Convert the final dictionary back to an argparse-like object (Namespace)
        args = argparse.Namespace(**config_dict)

    utils.create_directory_if_not_exists(f"./{exp_name}/outputs/")

    logger = utils.configure_logger(f"./{exp_name}/outputs/unlearn_{args.unlearn_method}.log")

    OUTPUT_CONFIG_FILE = f"./{exp_name}/outputs/unlearn_{args.unlearn_method}_config.yaml"
    OUTPUT_METRICS_FILE = f"./{exp_name}/outputs/unlearn_{args.unlearn_method}_metrics.yaml"
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    args.model_root = "/".join([".", exp_name, args.model_root])
    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, device_name = utils.device_configuration(args=args)
    logger.info(f"Unlearning scenario: {args.scenario} Dataset: {args.dataset} Unlearn method: {args.unlearn_method} Device: {device}")

    # Dataset
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=train_dataset,
        unlearn_class=args.unlearn_class
    )

    test_retain_dataset, _ = dataset.split_unlearn_dataset(
        data_list=test_dataset,
        unlearn_class=args.unlearn_class
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=args.batch_size, shuffle=True)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)
    unlearning_teacher = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)

    if args.unlearn_method != "retrain":
        # Load trained model to unlearn
        model.load_state_dict(torch.load(args.model_path))

    start_time = time.time()
    logger.info("Starting unlearning process...")
    # Unlearn
    unlearned_model = getattr(strategies, args.unlearn_method)(
        logger=logger,
        args=args,
        model=model,
        unlearning_teacher= unlearning_teacher,
        unlearn_class= args.unlearn_class,
        unlearn_loader=unlearn_loader,
        retain_loader=retain_loader,
        test_loader=test_loader,
        test_retain_loader=test_retain_loader,
        num_channels=num_channels,
        num_classes=num_classes,
        device=device
    )
    end_time = time.time()
    logger.info("Unlearning process completed.")
    runtime = end_time - start_time
    logger.info(f"Unlearned Runtime: {runtime}s")

    if args.save_model:
        utils.save_model(
            model=unlearned_model,
            model_name=args.unlearn_method,
            model_root=args.model_root,
        )

    # Evaluation after unlearning
    # Classification-level evaluation
    logger.info(f"Unlearned classification")
    retain_acc = metrics.evaluate(val_loader=retain_loader, model=unlearned_model, device=device)['Acc']
    logger.info(f"Retain acc: {retain_acc}")
    unlearn_acc = metrics.evaluate(val_loader=unlearn_loader, model=unlearned_model, device=device)['Acc']
    logger.info(f"Unlearn_acc: {unlearn_acc}")
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

    logger.info(f"Unlearned representation")
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

    linear_probe_acc = repr_metrics.linear_probing(
        train_loader= train_loader,
        retain_loader= retain_loader,
        forget_loader= unlearn_loader,
        model= unlearned_model,
        num_classes= num_classes,
        lr= args.linear_probe_lr,
    )
    logger.info(f"Linear probing acc: {linear_probe_acc}")
    
    if args.tsne:
        repr_metrics.visualize_tsne(
            reps=train_reps,
            all_labels=train_labels,
            unlearn_method=args.unlearn_method,
            exp_name=exp_name
        )
        logger.info("t-SNE visualization saved.")

    metrics_dict = {
        "classification/retain_acc": retain_acc,
        "classification/unlearn_acc": unlearn_acc,
        "classification/badt_mia": float(badt_mia),
        
        # attack model metrics
        "representation/badt_rep_mia": badt_rep_mia_metrics,
        "representation/pour_rmia": pour_rmia_metrics,
        "representation/sure_miars": sure_miars_metrics,
        
        # forget asr
        "representation/badt_rep_mia_asr": badt_rep_mia_asr,
        "representation/pour_rmia_asr": pour_rmia_asr,
        "representation/sure_miars_asr": sure_miars_asr,
        "representation/linear_probe_acc": linear_probe_acc,
        "runtime_sec": runtime
    }

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)