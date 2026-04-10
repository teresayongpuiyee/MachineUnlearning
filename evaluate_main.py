from src import utils
import argparse
from src import dataset, metrics, repr_metrics, analyse
from model import models
from unlearn_strategies import unlearn
from torch.utils.data import DataLoader
import yaml
import copy

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
# Unlearn configuration
parser.add_argument("-unlearn_class", type= int, help= "Class to unlearn")
parser.add_argument("-project_method", type= str, default= "", help= "Projection method for representation alignment",
                    choices=["orthogonal", "parallel", ""])

# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()


def main(args) -> None:
    unlearned_model_path_list = args.unlearned_model.split("/")
    exp_name = unlearned_model_path_list[-3]
    unlearn_method = unlearned_model_path_list[-1].split(".")[0]

    if len(args.project_method) > 0:
        output_path = f"./{exp_name}/evaluate_outputs_{args.project_method}/"
    else:
        output_path = f"./{exp_name}/evaluate_outputs/"
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

    # Get dataset info e.g., classes and channels
    num_classes, num_channels = dataset.dataset_info(dataset_name= args.dataset)

    unlearned_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    
    if unlearn_method == "pour_p":
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

    # Evaluation after unlearning
    if len(args.project_method) == 0:
        # Classification-level evaluation
        train_enp, train_enp_labels = metrics.get_entropy(train_loader, unlearned_model)
        test_enp, test_enp_labels = metrics.get_entropy(test_loader, unlearned_model)
        retain_enp, retain_enp_labels = metrics.get_entropy(retain_loader, unlearned_model)
        forget_enp, _ = metrics.get_entropy(unlearn_loader, unlearned_model)

        logger.info(f"Logit MIA evaluation...")
        ## Bad Teacher MIA
        #badt_mia = metrics.badt_mia(
        #    retain_loader=retain_loader,
        #    forget_loader=unlearn_loader,
        #    test_loader=test_loader,
        #    model=unlearned_model)
        #logger.info(f"Bad T MIA: {badt_mia}")

        # Bad Teacher equivalent MIA with balance and normalize features
        badt_mia_metrics, badt_mia_asr = repr_metrics.badt_rep_mia(
            retain_reps=retain_enp,
            forget_reps=forget_enp,
            test_reps=test_enp,
            retain_labels=retain_enp_labels,
            test_labels=test_enp_labels,
            unlearn_class=args.unlearn_class
        )
        logger.info(f"Bad T MIA: {badt_mia_asr}")

        # SCRUB equivalent MIA with balance and normalize features
        scrub_mia_metrics, scrub_mia_asr = repr_metrics.scrub_rep_mia(
            forget_reps=forget_enp,
            test_reps=test_enp,
            test_labels=test_enp_labels,
            unlearn_class=args.unlearn_class
        )
        logger.info(f"SCRUB MIA: {scrub_mia_asr}")

        # POUR
        pour_mia_metrics, pour_mia_asr = repr_metrics.pour_rmia(
            train_reps=train_enp,
            test_reps=test_enp,
            train_labels=train_enp_labels,
            test_labels=test_enp_labels,
            unlearn_class=args.unlearn_class,
        )
        logger.info(f"POUR MIA: {pour_mia_asr}")

        # SURE
        sure_mia_metrics, sure_mia_asr = repr_metrics.sure_miars(
            train_reps=train_enp,
            test_reps=test_enp,
            train_labels=train_enp_labels,
            test_labels=test_enp_labels,
            unlearn_class=args.unlearn_class,
        )
        logger.info(f"SURE MIA: {sure_mia_asr}")

        cls_metrics_dict = {
            # attack model metrics
            "badt_mia": badt_mia_metrics,
            "scrub_mia": scrub_mia_metrics,
            "pour_mia": pour_mia_metrics,
            "sure_mia": sure_mia_metrics,
            
            # forget asr
            "badt_mia_asr": badt_mia_asr,
            "scrub_mia_asr": scrub_mia_asr,
            "pour_mia_asr": pour_mia_asr,
            "sure_mia_asr": sure_mia_asr,
        }

    # Representation-level evaluation
    train_reps, train_labels = repr_metrics.get_representations(train_loader, unlearned_model)
    test_reps, test_labels = repr_metrics.get_representations(test_loader, unlearned_model)
    retain_reps, retain_labels = repr_metrics.get_representations(retain_loader, unlearned_model)
    forget_reps, _ = repr_metrics.get_representations(unlearn_loader, unlearned_model)

    if len(args.project_method) > 0:
        model_dir = "/".join(args.unlearned_model.split("/")[:-1])

        ori_model_path = model_dir + "/baseline.pt"
        ori_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=ori_model, model_path=ori_model_path,device=device)
        
        retrain_model_path = model_dir + "/retrain.pt"
        retrain_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=retrain_model, model_path=retrain_model_path,device=device)

        train_reps = analyse.project_representations(train_reps, ori_model, retrain_model, train_loader, device, projection=args.project_method)
        test_train_reps = analyse.project_representations(test_reps, ori_model, retrain_model, train_loader, device, projection=args.project_method)
        test_retain_reps = analyse.project_representations(test_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
        retain_reps = analyse.project_representations(retain_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
        forget_retain_reps = analyse.project_representations(forget_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
        forget_unlearn_reps = analyse.project_representations(forget_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)
        test_unlearn_reps = analyse.project_representations(test_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)       
    else:
        test_train_reps = test_reps
        test_retain_reps = test_reps
        forget_retain_reps = forget_reps       
        forget_unlearn_reps = forget_reps
        test_unlearn_reps = test_reps


    logger.info(f"Representation MIA evaluation...")
    # Bad Teacher equivalent Rep-MIA with balance and normalize features
    badt_rep_mia_metrics, badt_rep_mia_asr = repr_metrics.badt_rep_mia(
        retain_reps=retain_reps,
        forget_reps=forget_retain_reps,
        test_reps=test_retain_reps,
        retain_labels=retain_labels,
        test_labels=test_labels,
        unlearn_class=args.unlearn_class
    )
    logger.info(f"Bad T rep-MIA: {badt_rep_mia_asr}")

    # SCRUB equivalent Rep-MIA with balance and normalize features
    scrub_rep_mia_metrics, scrub_rep_mia_asr = repr_metrics.scrub_rep_mia(
        forget_reps=forget_unlearn_reps,
        test_reps=test_unlearn_reps,
        test_labels=test_labels,
        unlearn_class=args.unlearn_class
    )
    logger.info(f"SCRUB rep-MIA: {scrub_rep_mia_asr}")

    # POUR
    pour_rmia_metrics, pour_rmia_asr = repr_metrics.pour_rmia(
        train_reps=train_reps,
        test_reps=test_train_reps,
        train_labels=train_labels,
        test_labels=test_labels,
        unlearn_class=args.unlearn_class,
    )
    logger.info(f"POUR rMIA: {pour_rmia_asr}")

    # SURE
    sure_miars_metrics, sure_miars_asr = repr_metrics.sure_miars(
        train_reps=train_reps,
        test_reps=test_train_reps,
        train_labels=train_labels,
        test_labels=test_labels,
        unlearn_class=args.unlearn_class,
    )
    logger.info(f"SURE MIARS: {sure_miars_asr}")

    rep_metrics_dict = {       
        # attack model metrics
        "badt_rep_mia": badt_rep_mia_metrics,
        "scrub_rep_mia": scrub_rep_mia_metrics,
        "pour_rmia": pour_rmia_metrics,
        "sure_miars": sure_miars_metrics,
        
        # forget asr
        "badt_rep_mia_asr": badt_rep_mia_asr,
        "scrub_rep_mia_asr": scrub_rep_mia_asr,
        "pour_rmia_asr": pour_rmia_asr,
        "sure_miars_asr": sure_miars_asr,
    }
    
    if len(args.project_method) == 0:
        repr_metrics.visualize_tsne(
            reps=train_reps,
            all_labels=train_labels,
            unlearn_method=unlearn_method,
            exp_name=exp_name
        )
        logger.info("t-SNE visualization saved.")

    logger.info(f"Representation similarity evaluation...")
    # CKA
    if len(args.project_method) == 0:
        model_dir = "/".join(args.unlearned_model.split("/")[:-1])

        ori_model_path = model_dir + "/baseline.pt"
        ori_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=ori_model, model_path=ori_model_path,device=device)
        
        retrain_model_path = model_dir + "/retrain.pt"
        retrain_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(model=retrain_model, model_path=retrain_model_path,device=device)

    retain_ori_reps, _ = repr_metrics.get_representations(retain_loader, ori_model)
    forget_ori_reps, _ = repr_metrics.get_representations(unlearn_loader, ori_model)
    retain_retrain_reps, _ = repr_metrics.get_representations(retain_loader, retrain_model)
    forget_retrain_reps, _ = repr_metrics.get_representations(unlearn_loader, retrain_model)

    if len(args.project_method) > 0:
        retain_ori_reps = analyse.project_representations(retain_ori_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
        forget_ori_reps = analyse.project_representations(forget_ori_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)
        retain_retrain_reps = analyse.project_representations(retain_retrain_reps, ori_model, retrain_model, retain_loader, device, projection=args.project_method)
        forget_retrain_reps = analyse.project_representations(forget_retrain_reps, ori_model, retrain_model, unlearn_loader, device, projection=args.project_method)        
    
    cka_f_o = repr_metrics.linear_cka(forget_unlearn_reps, forget_ori_reps)
    cka_r_o = repr_metrics.linear_cka(retain_reps, retain_ori_reps)
    logger.info(f"CKA between unlearned and original model: forget={cka_f_o}, retain={cka_r_o}")

    cka_f_r = repr_metrics.linear_cka(forget_unlearn_reps, forget_retrain_reps)
    cka_r_r = repr_metrics.linear_cka(retain_reps, retain_retrain_reps)
    logger.info(f"CKA between unlearned and retrained model: forget={cka_f_r}, retain={cka_r_r}")

    cka_metrics_dict = {
        "forget_unlearn_original": cka_f_o,
        "retain_unlearn_original": cka_r_o,
        "forget_unlearn_retrain": cka_f_r,
        "retain_unlearn_retrain": cka_r_r,
    }

    # RUS
    rus_o = repr_metrics.representation_unlearning_score(cka_f_o, cka_r_o, original=True)
    logger.info(f"Representation Unlearning Score (RUS) with original model: {rus_o}")
    
    rus_r = repr_metrics.representation_unlearning_score(cka_f_r, cka_r_r)
    logger.info(f"Representation Unlearning Score (RUS) with retrained model: {rus_r}")

    rus_metrics_dict = {
        "unlearn_original": rus_o,
        "unlearn_retrain": rus_r,
    }

    metrics_dict = {
        "classification": cls_metrics_dict,
        "representation": rep_metrics_dict,
        "cka": cka_metrics_dict,
        "rus": rus_metrics_dict,
    }

    logger.info("Saving computed metrics...")
    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)