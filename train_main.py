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
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

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
parser.add_argument("-num_workers", type= int, default= 2, help= "Number of worker threads for data loading")
# Model
parser.add_argument("-model_root", type= str, default= "checkpoint", help= "Model root directory")
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
parser.add_argument("-pretrained_timm", dest="pretrained_timm", action="store_true", default=False, help="Use pretrained timm model")
parser.add_argument("-pretrained_weight", type= str, default= "", help= "Pretrained model path")
parser.add_argument("-resume", dest="resume", action="store_true", default=False, help="Resume training from checkpoint")
parser.add_argument("-model_name", type= str, default= "baseline", help= "Save model name")

# Training hyperparameter
parser.add_argument("-epochs", type= int, default= 30, help= "Training epochs")
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-lr", type=float, nargs='+', default= [1e-4], help='Learning rate(s)')
parser.add_argument("-optimizer", type= str, default= "adam", choices= ["sgd", "adam", "adamw"])
parser.add_argument('-momentum', type=float, default= 0.5, help='SGD momentum (default: 0.5)')
parser.add_argument('-weight_decay', type=float, default= 1e-4, help='Weight decay')
parser.add_argument("-nesterov", dest="nesterov", action="store_true", default=False, help="SGD nesterov momentum")
parser.add_argument('-label_smoothing', type=float, default=0.0, help='Label smoothing factor')
parser.add_argument("-scenario", type= str, default= "class",
                    choices= ["class", "client", "sample"], help= "Training and unlearning scenario")

parser.add_argument("-lr_scheduler", type= str, default= "constant", 
                    choices= [
                        "constant", 
                        "cosineannealingwarmrestarts",
                        "multisteplr",
                        "reducelronplateau",
                        "cosineannealing",
                        "exponential",
                        "step",
                        "polynomial"
                        ])
parser.add_argument("-milestones", type= int, nargs='+', default= [10, 20], help= "Steps for lr decay in multisteplr")
parser.add_argument("-t0", type= int, default= 5, help= "Number of epochs for the first restart in CosineAnnealingWarmRestarts")
parser.add_argument('-lr_patience', type=int, default=5, help='Learning plateau patience')
parser.add_argument('-lr_gamma', type=float, default=0.1, help='Learning rate decay factor')
parser.add_argument('-lr_factor', type=float, default=0.1, help='Learning rate factor for ReduceLROnPlateau')
parser.add_argument('-warm', type=int, default=0, help='Warm up training phase')
parser.add_argument("-lr_step_size", type= int, default= 15, help= "Step size for StepLR")
parser.add_argument('-min_lr', type=float, default=1e-6, help='Minimum learning rate')
parser.add_argument('-lr_power', type=float, default=0.9, help='Power for PolynomialLR')

parser.add_argument("-early_stop", dest="early_stop", action="store_true", default=False, help="Enable early stopping")
parser.add_argument("-mixup", dest="mixup", action="store_true", default=False, help="Enable mixup")
parser.add_argument('-es_patience', type=int, default=10, help='Early stopping patience')

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

    if os.path.exists(f"./{args.exp_name}") and not args.resume and len(args.pretrained_weight) == 0:
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

        if len(args.pretrained_weight) != 0:
            wandb_name = f"{args.exp_name}_pretrained"
        else:
            wandb_name = args.exp_name

        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project=args.project,
            name=wandb_name,
            dir="./"+args.exp_name,
            # Track hyperparameters and run metadata
            config=flattened_config,
        )
    args.model_root = "/".join([".", args.exp_name, args.model_root])

    # Set seed
    utils.set_seed(seed= args.seed)

    # Device
    device, device_name = utils.device_configuration(args= args)

    # Get dataset info e.g., classes and channels
    num_classes, num_channels = dataset.dataset_info(dataset_name= args.dataset)

    # Model preparation
    model = getattr(models, args.model)(
        num_classes= num_classes, input_channels= num_channels, pretrained=args.pretrained_timm).to(device)
    if args.pretrained_timm:
        logger.info("Using pretrained model from timm...")

    # Dataset
    train_dataset, test_dataset = dataset.get_dataset(
        dataset_name= args.dataset, root= args.root, model=model, pretrained_timm= args.pretrained_timm
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset, 
        batch_size= args.batch_size, 
        shuffle= True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=utils.seed_worker,
        generator=g
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle= False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Load pretrained weights if provided
    model = utils.load_pretrained_weights(
        model= model,
        pretrained_weight= args.pretrained_weight,
        device= device,
        logger= logger,
    )

    if args.optimizer not in ["sgd", "adam", "adamw"]:
        raise Exception("select correct optimizer")

    if len(args.lr) == 2:
        backbone_params = []
        fc_params = []

        for name, param in model.named_parameters():
            if name.startswith("fc."):
                fc_params.append(param)
            else:
                backbone_params.append(param)
        
        optim_param = [
            {"params": backbone_params, "lr": args.lr[0]},
            {"params": fc_params, "lr": args.lr[1]}
        ]
    elif len(args.lr) == 1:
        optim_param = [
            {"params": model.parameters(), "lr": args.lr[0]}
        ]
    else:
        raise ValueError("Invalid learning rate configuration. Accept a list of one or two floats.")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(optim_param, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(optim_param, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(optim_param, weight_decay=args.weight_decay)

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

    if args.warm > 0:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = scheduler.WarmUpLR(optimizer, iter_per_epoch * args.warm)
    else:
        warmup_scheduler = None

    if args.mixup:
        mixup_fn = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=0.5,
            prob=1.0,              # probability of applying
            switch_prob=0.3,       # mixup vs cutmix
            mode='batch',          # apply to whole batch
            label_smoothing=args.label_smoothing,
            num_classes=num_classes
        )
        loss_func = SoftTargetCrossEntropy().to(device)
    else:
        loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    if args.resume:
        (
            start_epoch, 
            patience_counter, 
            best_metrics, 
            best_test_acc, 
            best_test_loss
        ) = utils.load_checkpoint_and_resume(
            model, 
            optimizer, 
            lr_scheduler,
            f"{args.model_root}/{args.model_name}.pt",
            device
        )
    else:
        best_test_loss = float('inf')
        best_test_acc = -float('inf')
        best_metrics = dict()
        patience_counter = 0
        start_epoch = 1

    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        loss_list = []
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            if args.mixup:
                images, labels = mixup_fn(images, labels)

            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Evaluation preparation
            loss_list.append(loss.item())

            if warmup_scheduler is not None and epoch <= args.warm:
                warmup_scheduler.step()

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(val_loader= train_loader, model= model, device= device)['Acc']
        test_metrics = metrics.evaluate(val_loader= test_loader, model= model, device= device)
        test_loss = test_metrics['Loss']
        test_acc = test_metrics['Acc']

        if lr_scheduler is not None and epoch >= args.warm:
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
            for i, param_group in enumerate(optimizer.param_groups):
                metrics_dict.update({
                    f"lr_{i}": param_group['lr'],
                })
            wandb.log(metrics_dict)

        if args.early_stop:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_metrics = metrics_dict

            utils.save_model(
                checkpoint={
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'best_val_acc_metric': best_metrics,
                    'best_val_loss': best_test_loss,
                    'patience_counter': patience_counter
                },
                model_name=args.model_name,
                model_root=args.model_root,
            )

        if args.early_stop:
            if patience_counter >= args.es_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.dump(best_metrics, f, default_flow_style=False)