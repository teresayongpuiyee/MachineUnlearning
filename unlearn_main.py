from src import utils
import argparse
from src import dataset, metrics, repr_metrics
from model import models
from torch.utils.data import DataLoader
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

    train_eval_dataset, _, _, _ = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root, augment=False
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        dataset=train_dataset,
        unlearn_class=args.unlearn_class
    )

    retain_eval_dataset, unlearn_eval_dataset = dataset.split_unlearn_dataset(
        dataset=train_eval_dataset,
        unlearn_class=args.unlearn_class
    )

    test_retain_dataset, _ = dataset.split_unlearn_dataset(
        dataset=test_dataset,
        unlearn_class=args.unlearn_class
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_retain_loader = DataLoader(test_retain_dataset, batch_size=args.batch_size, shuffle=False)
    retain_eval_loader = DataLoader(retain_eval_dataset, batch_size=args.batch_size, shuffle=False)
    unlearn_eval_loader = DataLoader(unlearn_eval_dataset, batch_size=args.batch_size, shuffle=False)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)
    unlearning_teacher = getattr(models, args.model)(
        num_classes=num_classes, input_channels=num_channels).to(device)

    if args.unlearn_method != "retrain":
        # Load trained model to unlearn
        utils.load_model_weights(
            model=model,
            model_path=args.model_path,
            device=device
        )

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
            checkpoint=unlearned_model.state_dict(),
            model_name=args.unlearn_method,
            model_root=args.model_root,
        )

    # Evaluation after unlearning
    # Classification-level evaluation
    logger.info(f"Unlearned classification")
    retain_acc = metrics.evaluate(val_loader=retain_eval_loader, model=unlearned_model, device=device)['Acc']
    logger.info(f"Retain acc: {retain_acc}")
    unlearn_acc = metrics.evaluate(val_loader=unlearn_eval_loader, model=unlearned_model, device=device)['Acc']
    logger.info(f"Unlearn_acc: {unlearn_acc}")

    logger.info(f"Unlearned representation")
    linear_probe_acc = repr_metrics.linear_probing(
        train_loader= train_loader,
        retain_eval_loader= retain_eval_loader,
        unlearn_eval_loader= unlearn_eval_loader,
        model= unlearned_model,
        num_classes= num_classes,
        lr= args.linear_probe_lr,
    )
    logger.info(f"Linear probing acc: {linear_probe_acc}")

    metrics_dict = {
        "classification/retain_acc": retain_acc,
        "classification/unlearn_acc": unlearn_acc,
        "representation/linear_probe_acc": linear_probe_acc,
        "runtime_sec": runtime
    }

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)