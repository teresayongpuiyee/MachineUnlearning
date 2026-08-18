from src import utils
import argparse
from src import dataset, metrics, repr_metrics, analyse
from model import models
from unlearn_strategies import unlearn
from torch.utils.data import DataLoader
import yaml
import copy
import re
import csv

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
parser.add_argument("-pretrained_timm", dest="pretrained_timm", action="store_true", default=False, help="Model trained with pretrained timm")
# Model
parser.add_argument("-model", type= str, default= "ResNet18", help= "Model selection")
parser.add_argument("-unlearned_model", type=str, required=True, help="Path to unlearned model")
parser.add_argument("-retrain_model_name", type= str, default= "retrain", help= "Retrain model name")
parser.add_argument("-ori_model_name", type= str, default= "baseline", help= "Original model name")

# Unlearn configuration
parser.add_argument("-unlearn_class", type= int, help= "Class to unlearn")
parser.add_argument("-project_method", type= str, default= "", help= "Projection method for representation alignment",
                    choices=["orthogonal", "parallel", ""])
parser.add_argument("-metrics", type= str, nargs='+', 
                    default= ["mia_logit", 
                              "mia_rep", 
                              "cka_o",
                              "cka_r",
                              "tsne",
                              "relearn_attack",
                              "svcca",
                              "rand_proj"
                              ], 
                    help= "Metrics to evaluate")
parser.add_argument("-num_rand", type= int, default= 50, help= "Number of random directions")

# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
parser.add_argument("-relearn_lr", type=float, default= 1e-3, help='Learning rate')
parser.add_argument("-relearn_momentum", type=float, default= 0.0, help='Momentum')
parser.add_argument("-relearn_wd", type=float, default= 0.0, help='Weight decay')
parser.add_argument("-relearn_epoch", type=int, default= 30, help='Epoch')
parser.add_argument("-sample_size", type=int, default= 5, help='Sample size')
parser.add_argument("-retain_per_forget", type=int, default= 0, help='Number of retain class')

# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()


def main(args) -> None:
    unlearned_model_path_list = args.unlearned_model.split("/")
    exp_name = unlearned_model_path_list[-4]
    unlearn_method = unlearned_model_path_list[-1].split(".")[0]

    if len(args.project_method) > 0:
        output_path = f"./{exp_name}/{args.unlearn_class}/evaluate_outputs_{args.project_method}/"
    else:
        output_path = f"./{exp_name}/{args.unlearn_class}/evaluate_outputs/"
    utils.create_directory_if_not_exists(output_path)
    
    logger = utils.configure_logger(f"{output_path}unlearn_{unlearn_method}_{args.ori_model_name}_{args.retrain_model_name}.log")
    OUTPUT_CONFIG_FILE = f"{output_path}unlearn_{unlearn_method}_{args.ori_model_name}_{args.retrain_model_name}_config.yaml"
    OUTPUT_METRICS_FILE = f"{output_path}unlearn_{unlearn_method}_{args.ori_model_name}_{args.retrain_model_name}_metrics.yaml"
    
    config_dict = vars(args).copy()
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logger.info(f"Evaluating unlearning method {unlearn_method} on dataset {args.dataset}")
    
    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, _ = utils.device_configuration(args=args)

    # Get dataset info e.g., classes and channels
    num_classes, num_channels = dataset.dataset_info(dataset_name= args.dataset)

    unlearned_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    
    if re.sub(r"\d+", "", unlearn_method) == "pour_p":
        unlearned_model = unlearn.POUR_P(
            copy.deepcopy(unlearned_model), 
            args.unlearn_class
        ).to(device)
    
    # Dataset
    logger.info("Preparing datasets and dataloaders...")
    train_dataset, test_dataset = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root, augment=False, model=unlearned_model, pretrained_timm= args.pretrained_timm
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        dataset=train_dataset,
        unlearn_class=args.unlearn_class
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # Model preparation
    logger.info("Loading model checkpoints...")
    utils.load_model_weights(model=unlearned_model, model_path=args.unlearned_model,device=device)

    cls_metrics_dict = dict()
    rep_metrics_dict = dict()
    cka_r_metrics_dict = dict()
    cka_o_metrics_dict = dict()
    shift_norm_dict = dict()
    svcca_metrics_dict = dict()
    mia_sum = dict()
    cka_f_sum = dict()

    # Evaluation after unlearning
    if "mia_logit" in args.metrics and len(args.project_method) == 0:
        # Classification-level evaluation
        train_enp, train_enp_labels = metrics.get_entropy(train_loader, unlearned_model)
        test_enp, test_enp_labels = metrics.get_entropy(test_loader, unlearned_model)

        logger.info(f"Logit MIA evaluation...")
        # POUR
        pour_mia_metrics, pour_mia_asr = repr_metrics.pour_rmia(
            train_reps=train_enp,
            test_reps=test_enp,
            train_labels=train_enp_labels,
            test_labels=test_enp_labels,
            unlearn_class=args.unlearn_class,
        )
        logger.info(f"POUR MIA: {pour_mia_asr}")

        cls_metrics_dict = {
            # attack model metrics
            "pour_mia": pour_mia_metrics,
            # forget asr
            "pour_mia_asr": pour_mia_asr,
        }

    # Representation-level evaluation
    if ("cka_o" in args.metrics or "cka_r" in args.metrics or "svcca" in args.metrics or "mia_rep" in args.metrics) and len(args.project_method) > 0:
        model_dir = "/".join(args.unlearned_model.split("/")[:-1])

        ori_model_path = model_dir + f"/{args.ori_model_name}.pt"
        ori_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=ori_model, model_path=ori_model_path,device=device)
        
        retrain_model_path = model_dir + f"/{args.retrain_model_name}.pt"
        retrain_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=retrain_model, model_path=retrain_model_path,device=device)
    
    if "mia_rep" in args.metrics or "tsne" in args.metrics:
        train_reps, train_labels = repr_metrics.get_representations(train_loader, unlearned_model)
        test_reps, test_labels = repr_metrics.get_representations(test_loader, unlearned_model)

        raw_train_reps = train_reps
        raw_test_reps = test_reps

        if len(args.project_method) > 0:
            train_reps, train_shift_norm = analyse.project_representations(train_reps, ori_model, retrain_model, train_loader, device, projection=args.project_method)
            test_reps, _ = analyse.project_representations(test_reps, ori_model, retrain_model, train_loader, device, projection=args.project_method)

            shift_norm_dict.update({
                "train": train_shift_norm
            })

    if "mia_rep" in args.metrics:
        logger.info(f"Representation MIA evaluation...")

        # POUR
        pour_rmia_metrics, pour_rmia_asr = repr_metrics.pour_rmia(
            train_reps=train_reps,
            test_reps=test_reps,
            train_labels=train_labels,
            test_labels=test_labels,
            unlearn_class=args.unlearn_class,
        )
        logger.info(f"POUR rMIA: {pour_rmia_asr}")

        rep_metrics_dict = {       
            # attack model metrics
            "pour_rmia": pour_rmia_metrics,
            # forget asr
            "pour_rmia_asr": pour_rmia_asr,
        }

    if "tsne" in args.metrics and len(args.project_method) == 0 and num_classes <= 20:  # Only visualize when not projecting and number of classes is manageable
        repr_metrics.visualize_tsne(
            reps=train_reps,
            all_labels=train_labels,
            unlearn_method=unlearn_method,
            save_path=output_path, 
            tag="rep",
        )
        logger.info("t-SNE visualization saved.")

    if "cka_o" in args.metrics or "cka_r" in args.metrics or "svcca" in args.metrics:
        retain_reps, _ = repr_metrics.get_representations(retain_loader, unlearned_model)
        forget_reps, _ = repr_metrics.get_representations(unlearn_loader, unlearned_model)

        raw_forget_reps = forget_reps

        if len(args.project_method) > 0:
            retain_reps, retain_shift_norm = analyse.project_representations(retain_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
            forget_reps, forget_shift_norm = analyse.project_representations(forget_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)

            shift_norm_dict.update({
                "retain": retain_shift_norm,
                "forget": forget_shift_norm
            })

    if "cka_o" in args.metrics:
        logger.info(f"Representation similarity evaluation with original model...")
        
        if len(args.project_method) == 0:
            model_dir = "/".join(args.unlearned_model.split("/")[:-1])
            ori_model_path = model_dir + f"/{args.ori_model_name}.pt"
            ori_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
            utils.load_model_weights(model=ori_model, model_path=ori_model_path,device=device)

        retain_ori_reps, _ = repr_metrics.get_representations(retain_loader, ori_model)
        forget_ori_reps, _ = repr_metrics.get_representations(unlearn_loader, ori_model)
        
        if len(args.project_method) > 0:
            retain_ori_reps, _ = analyse.project_representations(retain_ori_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
            forget_ori_reps, _ = analyse.project_representations(forget_ori_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)
        
        cka_f_o = repr_metrics.linear_cka(forget_reps, forget_ori_reps)
        cka_r_o = repr_metrics.linear_cka(retain_reps, retain_ori_reps)
        logger.info(f"CKA between unlearned and original model: forget={cka_f_o}, retain={cka_r_o}")

        rus_o = repr_metrics.representation_unlearning_score(cka_f_o, cka_r_o, original=True)
        logger.info(f"Representation Unlearning Score (RUS) with original model: {rus_o}")

        cka_o_metrics_dict = {
            "forget_unlearn_original": cka_f_o,
            "retain_unlearn_original": cka_r_o,
            "rus_unlearn_original": rus_o,
        }

    if "cka_r" in args.metrics or "svcca" in args.metrics:
        logger.info(f"Representation similarity evaluation with retrained model...")
        #if len(args.project_method) == 0:
        model_dir = "/".join(args.unlearned_model.split("/")[:-1])

        reference_model_path = model_dir + "/retrain.pt"
        reference_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=reference_model, model_path=reference_model_path,device=device)

        retain_retrain_reps, _ = repr_metrics.get_representations(retain_loader, reference_model)
        forget_retrain_reps, _ = repr_metrics.get_representations(unlearn_loader, reference_model)

        raw_forget_retrain_reps = forget_retrain_reps

        if len(args.project_method) > 0:
            retain_retrain_reps, _ = analyse.project_representations(retain_retrain_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
            forget_retrain_reps, _ = analyse.project_representations(forget_retrain_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)   

        if "cka_r" in args.metrics:
            cka_f_r = repr_metrics.linear_cka(forget_reps, forget_retrain_reps)
            cka_r_r = repr_metrics.linear_cka(retain_reps, retain_retrain_reps)
            logger.info(f"CKA between unlearned and retrained model: forget={cka_f_r}, retain={cka_r_r}")

            rus_r = repr_metrics.representation_unlearning_score(cka_f_r, cka_r_r)
            logger.info(f"Representation Unlearning Score (RUS) with retrained model: {rus_r}")

            cka_r_metrics_dict = {
                "forget_unlearn_retrain": cka_f_r,
                "retain_unlearn_retrain": cka_r_r,
                "rus_unlearn_retrain": rus_r,
            }

        if "svcca" in args.metrics:
            svcca_f_r, _ = repr_metrics.svcca(forget_reps, forget_retrain_reps)
            svcca_r_r, _ = repr_metrics.svcca(retain_reps, retain_retrain_reps)

            svcca_metrics_dict = {
                "forget_unlearn_retrain": svcca_f_r,
                "retain_unlearn_retrain": svcca_r_r,
            }

    if "relearn_attack" in args.metrics:
        repr_metrics.relearning_attack(
            logger,
            unlearned_model,
            unlearn_loader,
            retain_loader,
            test_loader,
            sample_size=args.sample_size,
            epoch=args.relearn_epoch,
            lr=args.relearn_lr,
            momentum=args.relearn_momentum,
            weight_decay=args.relearn_wd,
            device=device,
            seed=args.seed,
            model_name=unlearn_method,
            save_dir=output_path,
            retain_per_forget=args.retain_per_forget
        )

    if "rand_proj" in args.metrics and ("mia_rep" in args.metrics or "cka_r" in args.metrics) and len(args.project_method) > 0:
        M = args.num_rand

        mia_rep_proj = pour_rmia_asr
        cka_f_r_proj = cka_f_r

        null_cka_f, null_mia = [], []
        for s in range(M):

            train_random_reps, _ = analyse.project_representations(raw_train_reps, None, None, None, device, projection=args.project_method, random=True, seed=s)
            test_random_reps, _ = analyse.project_representations(raw_test_reps, None, None, None, device, projection=args.project_method, random=True, seed=s)
            
            _, pour_rand_rmia_asr = repr_metrics.pour_rmia(
                train_reps=train_random_reps,
                test_reps=test_random_reps,
                train_labels=train_labels,
                test_labels=test_labels,
                unlearn_class=args.unlearn_class,
            )
            null_mia.append(pour_rand_rmia_asr)

            forget_rand_reps, _ = analyse.project_representations(raw_forget_reps, None, None, None, device, projection=args.project_method, random=True, seed=s)
            forget_rand_retrain_reps, _ = analyse.project_representations(raw_forget_retrain_reps, None, None, None, device, projection=args.project_method, random=True, seed=s)   

            cka_f_r_rand = repr_metrics.linear_cka(forget_rand_reps, forget_rand_retrain_reps)

            null_cka_f.append(cka_f_r_rand)

        # write null_cka_f, null_mia to csv file
        csv_path = f"{output_path}{unlearn_method}_random.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["null_cka_f", "null_mia"])
            writer.writerows(zip(null_cka_f, null_mia))
        logger.info(f"Saved null distributions to {csv_path}")

        mia_sum = analyse.summarize_against_null(mia_rep_proj,  null_mia)
        cka_f_sum = analyse.summarize_against_null(cka_f_r_proj,  null_cka_f)

    metrics_dict = {
        "classification": cls_metrics_dict,
        "representation": rep_metrics_dict,
        "cka_retrain": cka_r_metrics_dict,
        "cka_original": cka_o_metrics_dict,
        "svcca_retrain": svcca_metrics_dict,
        "shift_norm": shift_norm_dict,
        "mia_sum": mia_sum,
        "cka_f_sum": cka_f_sum
    }

    logger.info("Saving computed metrics...")
    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)