"""
Unlearning utility file
"""
from torch.utils.data import DataLoader
import torch
import copy
from torch import nn
from tqdm import tqdm
import numpy as np
from src import metrics
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
    args: argparse.Namespace = None
) -> torch.nn.Module:
    # Copy model, avoid overwriting
    trained_model = copy.deepcopy(model)

    if desc == "Retraining model":
        epochs = args.epochs
        opt = args.optimizer
        early_stop = args.early_stop
        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        patience = args.patience
        best_test_loss = float('inf')
        best_test_acc = -float('inf')
        patience_counter = 0
    else:
        lr = 1e-4
        momentum = 0.5
        weight_decay = 1e-4

    if opt not in ["sgd", "adam"]:
        raise Exception("Select correct optimizer")
    if opt == "sgd":
        optimizer = torch.optim.SGD(trained_model.parameters(), lr=lr, momentum= momentum)
    else:
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=lr, weight_decay=weight_decay)

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
        logger.info( f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc} Test acc: {test_acc}")

        if desc == "Retraining model":
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_trained_model = trained_model

            if early_stop:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        else:
            best_trained_model = trained_model

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