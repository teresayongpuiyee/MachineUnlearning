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
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy


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
    nesterov: bool = False,
    label_smoothing: float = 0.0,
    mixup: bool = False,
) -> torch.nn.Module:
    # Copy model, avoid overwriting
    trained_model = copy.deepcopy(model)

    if desc == "Retraining model":
        epochs = args.epochs
        opt = args.optimizer
        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        if hasattr(args, "nesterov"):
            nesterov = args.nesterov
        label_smoothing = args.label_smoothing
        best_test_loss = float('inf')
        best_test_acc = -float('inf')
        patience_counter = 0

        trained_model = utils.load_pretrained_weights(
            model= trained_model,
            pretrained_weight= args.retrain_pretrained_weight,
            device= device,
            logger= logger,
        )

        mixup = args.mixup

    if opt not in ["sgd", "adam", "adamw"]:
        raise Exception("Select correct optimizer")

    if isinstance(lr, list):
        if len(lr) == 2:
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
        elif len(lr) == 1:
            optim_param = [
                {"params": trained_model.parameters(), "lr": lr[0]}
            ]
    elif isinstance(lr, float):
        optim_param = [
            {"params": trained_model.parameters(), "lr": lr}
        ]
    else:
        raise ValueError("Invalid learning rate configuration. Accept a float or a list of at most two floats.")

    if opt == "sgd":
        optimizer = torch.optim.SGD(optim_param, momentum= momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif opt == "adam":
        optimizer = torch.optim.Adam(optim_param, weight_decay=weight_decay)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(optim_param, weight_decay=weight_decay)

    if desc == "Retraining model":
        if hasattr(args, "lr_scheduler"):
            lr_scheduler = scheduler.get_lr_scheduler(
                args.lr_scheduler, 
                optimizer, 
                milestones=args.milestones, 
                epochs=args.epochs - args.warm, 
                t0=args.t0,
                lr_patience=args.lr_patience,
                lr_gamma=args.lr_gamma,
                lr_factor=args.lr_factor,
                lr_step_size=args.lr_step_size,
                min_lr=args.min_lr,
                lr_power=args.lr_power
            )
        else:
            lr_scheduler = None

        if args.warm > 0:
            iter_per_epoch = len(train_loader)
            warmup_scheduler = scheduler.WarmUpLR(optimizer, iter_per_epoch * args.warm)
        else:
            warmup_scheduler = None

    if desc == "Retraining model" and mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=0.5,
            prob=1.0,              # probability of applying
            switch_prob=0.3,       # mixup vs cutmix
            mode='batch',          # apply to whole batch
            label_smoothing=label_smoothing,
            num_classes=len(train_loader.dataset.dataset.classes)
        )
        loss_func = SoftTargetCrossEntropy().to(device)
    else:
        loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    for epoch in tqdm(range(1, epochs + 1), desc= desc):
        loss_list = []
        #trained_model.train()
        trained_model.eval()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            if desc == "Retraining model" and mixup:
                images, labels = mixup_fn(images, labels)

            trained_model.zero_grad()
            output = trained_model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            if desc == "Retraining model":
                if warmup_scheduler is not None and epoch <= args.warm:
                    warmup_scheduler.step()

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(val_loader= train_loader, model= trained_model, device= device)['Acc']
        test_metrics = metrics.evaluate(val_loader= test_loader, model= trained_model, device= device)
        test_loss = test_metrics['Loss']
        test_acc = test_metrics['Acc']
        logger.info( f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Test Loss: {test_loss:.4f} Train Acc: {train_acc} Test acc: {test_acc}")

        if desc == "Retraining model":
            if lr_scheduler is not None and epoch >= args.warm:
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

                if patience_counter >= args.es_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            utils.save_model(
                checkpoint=trained_model.state_dict(),
                model_name=f"{args.model_name}{epoch}",
                model_root=args.model_root,
            )
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

def get_fc(module):
    if hasattr(module, "fc"):
        return module.fc
    elif hasattr(module, "head"):
        return module.head
    elif hasattr(module, "model") and hasattr(module.model, "fc"):
        return module.model.fc
    elif hasattr(module, "model") and hasattr(module.model, "head"):
        return module.model.head
    else:
        raise AttributeError("Cannot find fc or head layer")