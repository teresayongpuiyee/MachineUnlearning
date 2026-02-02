import copy
from src import dataset, metrics, utils, scheduler
import argparse
import torch
from torch.utils.data import DataLoader
from model import models
from torch import nn
from tqdm import tqdm
import numpy as np
import wandb
import datetime
import os
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
parser.add_argument("-model_root", type= str, default= "checkpoint", help= "Model root directory")
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
parser.add_argument("-pretrained_weight", type= str, default= "", help= "Pretrained model path")

# Training hyperparameter
parser.add_argument("-epochs", type= int, default= 30, help= "Training epochs")
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-lr", type=float, default= 1e-4, help='Learning rate')
parser.add_argument("-optimizer", type= str, default= "adam", choices= ["sgd", "adam"])
parser.add_argument('-momentum', type=float, default= 0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('-weight_decay', type=float, default= 1e-4, help='Weight decay')
parser.add_argument("-scenario", type= str, default= "class",
                    choices= ["class", "client", "sample"], help= "Training and unlearning scenario")

parser.add_argument("-lr_scheduler", type= str, default= "constant", 
                    choices= [
                        "constant", 
                        "cosineannealingwarmrestarts",
                        "multisteplr",
                        "reducelronplateau",
                        "cosineannealing"
                        ])
parser.add_argument("-milestones", type= int, nargs='+', default= [10, 20], help= "Steps for lr decay in multisteplr")
parser.add_argument("-t0", type= int, default= 5, help= "Number of epochs for the first restart in CosineAnnealingWarmRestarts")

parser.add_argument("-early_stop", dest="early_stop", action="store_true", default=False, help="Enable early stopping")
parser.add_argument('-patience', type=int, default=10, help='Early stopping patience')

# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

parser.add_argument("-project", default="machine_unlearning", type=str, help="wandb project name")
parser.add_argument("-exp_name", type=str, help="experiment name")
parser.add_argument("-wandb", dest="wandb", action="store_true", default=False, help="log in wandb")

args = parser.parse_args()
timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

def flatten_dict(d, prefix=''):
    """
    Recursively flatten a dictionary with dot notation keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":

    if os.path.exists(f"./{args.exp_name}"):
        args.exp_name = args.exp_name + "_" + timestamp

    utils.create_directory_if_not_exists(f"./{args.exp_name}/")

    logger = utils.configure_logger(f"./{args.exp_name}/train.log")

    OUTPUT_CONFIG_FILE = f"./{args.exp_name}/train_config.yaml"
    OUTPUT_METRICS_FILE = f"./{args.exp_name}/train_metrics.yaml"
    config_dict = vars(args)
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    if args.wandb:
        # Convert to OmegaConf object
        flattened_config = flatten_dict(config_dict)

        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project=args.project,
            name=args.exp_name,
            dir="./"+args.exp_name,
            # Track hyperparameters and run metadata
            config=flattened_config,
        )
    args.model_root = "/".join([".", args.exp_name, args.model_root])

    # Set seed
    utils.set_seed(seed= args.seed)

    # Device
    device, device_name = utils.device_configuration(args= args)

    # Dataset
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name= args.dataset, root= args.root
    )

    num_classes = num_classes - 1  # Adjust number of classes after unlearning

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=train_dataset,
        unlearn_class=0
    )

    test_retain_dataset, test_unlearn_dataset = dataset.split_unlearn_dataset(
        data_list=test_dataset,
        unlearn_class=0
    )

    train_loader = DataLoader(retain_dataset, batch_size= args.batch_size, shuffle= True)
    test_loader = DataLoader(test_retain_dataset, batch_size=args.batch_size, shuffle= True)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes= num_classes, input_channels= num_channels).to(device)

    model = utils.load_pretrained_weights(
        model= model,
        pretrained_weight= args.pretrained_weight,
        device= device,
        logger= logger,
    )

    if args.optimizer not in ["sgd", "adam"]:
        raise Exception("select correct optimizer")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = scheduler.get_lr_scheduler(
        args.lr_scheduler, 
        optimizer, 
        milestones=args.milestones, 
        epochs=args.epochs, 
        t0=args.t0
    )

    loss_func = nn.CrossEntropyLoss().to(device)
    best_test_loss = float('inf')
    best_test_acc = -float('inf')
    best_train_acc = -float('inf')
    patience_counter = 0
    start_epoch = 1

    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        loss_list = []
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Evaluation preparation
            loss_list.append(loss.item())

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(val_loader= train_loader, model= model, device= device)['Acc']
        test_metrics = metrics.evaluate(val_loader= test_loader, model= model, device= device)
        test_loss = test_metrics['Loss']
        test_acc = test_metrics['Acc']

        if lr_scheduler is not None:
            if args.lr_scheduler == "reducelronplateau":
                lr_scheduler.step(test_loss)
            else:
                lr_scheduler.step()

        metrics_dict = {
            "val/accuracy": test_acc, 
            "val/loss": test_loss,
            "train/accuracy": train_acc,
            "train/loss": float(mean_loss),
            "epoch": epoch
        }

        logger.info(f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc} Test Acc: {test_acc}")
        
        if args.wandb:
            cur_lr = optimizer.param_groups[0]['lr']
            metrics_dict.update({
                "lr": cur_lr,
            })
            wandb.log(metrics_dict)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)
            best_metrics = metrics_dict

        if args.early_stop:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    utils.save_model(
        model=best_model,
        model_name="baseline",
        model_root=args.model_root,
    )

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.dump(best_metrics, f, default_flow_style=False)