import copy
from src import dataset, metrics, utils
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
parser.add_argument("-save_model", type= bool, default= True, help= "Save trained model option")

# Training hyperparameter
parser.add_argument("-epochs", type= int, default= 30, help= "Training epochs")
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-lr", type=float, default= 1e-4, help='Learning rate')
parser.add_argument("-optimizer", type= str, default= "adam", choices= ["sgd, adam"])
parser.add_argument('-momentum', type=float, default= 0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('-weight_decay', type=float, default= 1e-4, help='Weight decay')
parser.add_argument("-scenario", type= str, default= "class",
                    choices= ["class", "client", "sample"], help= "Training and unlearning scenario")

parser.add_argument('-report_training', type= bool, default= False, help= "option to show training performance")
parser.add_argument('-report_interval', type= int, default= 5, help= "training performance report interval")

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

    if os.path.exists(os.path.dirname("./"+args.exp_name)):
        args.exp_name = args.exp_name + "_" + timestamp

    logger = utils.configure_logger(f"./{args.exp_name}/train.log")

    utils.create_directory_if_not_exists("./"+args.exp_name, logger)

    if args.wandb:
        # Convert to OmegaConf object
        config_dict = vars(args)
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

    train_loader = DataLoader(train_dataset, batch_size= args.batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle= True)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes= num_classes, input_channels= num_channels).to(device)

    if args.optimizer not in ["sgd", "adam"]:
        raise Exception("select correct optimizer")
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_func = nn.CrossEntropyLoss().to(device)
    best_test_loss = float('inf')
    best_test_acc = -float('inf')
    best_train_acc = -float('inf')
    patience_counter = 0

    for epoch in tqdm(range(1, args.epochs + 1)):
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
        if args.report_training:
            logger.info(f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc} Test Acc: {test_acc}")
        
        if args.wandb:
            cur_lr = optimizer.param_groups[0]['lr']

            wandb.log({
                "val/accuracy": test_acc, 
                "val/loss": test_loss,
                "train/accuracy": train_acc,
                "train/loss": mean_loss,
                "lr": cur_lr,
                "epoch": epoch
            })

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)

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
        train_acc=best_train_acc,
        test_acc=best_test_acc,
        logger=logger
    )