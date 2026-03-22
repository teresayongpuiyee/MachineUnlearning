"""
Unlearning utility file
"""
from torch.utils.data import DataLoader
import torch
import copy
from torch import nn
from tqdm import tqdm
import numpy as np
from src import metrics, scheduler, utils
import argparse
from typing import Tuple
import os


def training_optimization(
    logger,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    desc: str,
    opt: str= "adam",
    args: argparse.Namespace = None,
    lr: float = 1e-4,
    momentum: float = 0.5,
    weight_decay: float = 1e-4,
) -> torch.nn.Module:
    # Copy model, avoid overwriting
    trained_model = copy.deepcopy(model)

    if desc == "Retraining model":
        epochs = args.epochs
        opt = args.optimizer
        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        best_test_loss = float('inf')
        best_test_acc = -float('inf')
        patience_counter = 0

        trained_model = utils.load_pretrained_weights(
            model= trained_model,
            pretrained_weight= args.retrain_pretrained_weight,
            device= device,
            logger= logger,
        )

    if opt not in ["sgd", "adam"]:
        raise Exception("Select correct optimizer")

    if isinstance(lr, list) and len(lr) == 2:
        backbone_params = []
        fc_params = []

        for name, param in trained_model.named_parameters():
            if name.startswith("fc."):
                fc_params.append(param)
            else:
                backbone_params.append(param)
        
        optim_param = [
            {"params": backbone_params, "lr": lr[0]},
            {"params": fc_params, "lr": lr[1]}
        ]
    elif isinstance(lr, float):
        optim_param = [
            {"params": trained_model.parameters(), "lr": lr}
        ]
    else:
        raise ValueError("Invalid learning rate configuration. Accept a float or a list of two floats.")

    if opt == "sgd":
        optimizer = torch.optim.SGD(optim_param, momentum= momentum)
    else:
        optimizer = torch.optim.Adam(optim_param, weight_decay=weight_decay)

    if desc == "Retraining model":
        if hasattr(args, "lr_scheduler"):
            lr_scheduler = scheduler.get_lr_scheduler(
                args.lr_scheduler, 
                optimizer, 
                milestones=args.milestones, 
                epochs=args.epochs, 
                t0=args.t0
            )
        else:
            lr_scheduler = None

    loss_func = nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(1, epochs + 1), desc= desc):
        loss_list = []
        trained_model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            trained_model.zero_grad()
            output = trained_model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(val_loader= train_loader, model= trained_model, device= device)['Acc']
        test_metrics = metrics.evaluate(val_loader= test_loader, model= trained_model, device= device)
        test_loss = test_metrics['Loss']
        test_acc = test_metrics['Acc']
        logger.info( f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Test Loss: {test_loss:.4f} Train Acc: {train_acc} Test acc: {test_acc}")

        if desc == "Retraining model":
            if lr_scheduler is not None:
                if args.lr_scheduler == "reducelronplateau":
                    lr_scheduler.step(test_loss)
                else:
                    lr_scheduler.step()

            # Get retrain model with best test acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_trained_model = copy.deepcopy(trained_model)

            # To prevent overfitting
            if args.early_stop:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            best_trained_model = copy.deepcopy(trained_model)

    return best_trained_model


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


def create_directory_if_not_exists(
    file_path: str
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
    model_arc: str,
    model: torch.nn.Module,
    scenario: str,
    model_name: str,
    model_root: str,
    dataset_name: str,
    train_acc: float,
    test_acc: float,
) -> None:
    model_folder = f"{model_root}/{model_arc}/{scenario}/{dataset_name}/"
    create_directory_if_not_exists(file_path=model_folder)
    model_path = f"{model_folder}{model_name}_{train_acc}_{test_acc}.pt"
    torch.save(model.state_dict(), model_path)