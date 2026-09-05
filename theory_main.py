from src import utils, dataset, theory, repr_metrics, analyse
import argparse
from model import models
from torch.utils.data import DataLoader
import yaml
import torch

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
# Unlearn configuration
parser.add_argument("-unlearn_class", type= int, default=0, help= "Class to unlearn")
parser.add_argument("-model_dir", type=str, required=True, help="Path to models directory")
parser.add_argument("-exps", type= str, nargs='+', 
                    default= ["concentration", 
                              "finetune",
                              "gradient",
                              "forget_shift"
                              ], 
                    help= "Experiments to evaluate")
parser.add_argument("-finetune_mode", type= str, default= "eval", help= "Mode for finetuning: eval or train")
parser.add_argument("-retrain_model_name", type= str, default= "retrain", help= "Retrain model name")

# Training hyperparameter
parser.add_argument("-batch_size", type= int, default= 128, help= "Training batch size")
# Set seed
parser.add_argument("-seed", type=int,default= 0, help="Seed for runs")

args = parser.parse_args()


def main(args) -> None:
    model_dir_list = args.model_dir.split("/")
    exp_name = model_dir_list[-3]
    
    output_path = f"./{exp_name}/{args.unlearn_class}/theory_outputs/"
    utils.create_directory_if_not_exists(output_path)
    
    exp_type = "_".join(args.exps)
    logger = utils.configure_logger(f"{output_path}{exp_type}.log")
    OUTPUT_CONFIG_FILE = f"{output_path}{exp_type}_config.yaml"
    OUTPUT_METRICS_FILE = f"{output_path}{exp_type}_metrics.yaml"

    config_dict = vars(args).copy()
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Set seed
    utils.set_seed(seed=args.seed)

    # Device
    device, _ = utils.device_configuration(args=args)

    # Get dataset info e.g., classes and channels
    num_classes, num_channels = dataset.dataset_info(dataset_name= args.dataset)

    unlearned_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)

    # Dataset
    logger.info("Preparing datasets and dataloaders...")
    train_dataset, _ = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root, augment=False, model=unlearned_model, pretrained_timm= args.pretrained_timm
    )

    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        dataset=train_dataset,
        unlearn_class=args.unlearn_class
    )

    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # Model preparation
    logger.info("Loading original model checkpoints...")
    ori_model_path = f"{args.model_dir}/baseline.pt"
    ori_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    utils.load_model_weights(ori_model, ori_model_path, device)

    if "gradient" in args.exps or "concentration" in args.exps or "forget_shift" in args.exps:
        logger.info("Loading retrained model checkpoints...")
        retrain0_model_path = f"{args.model_dir}/retrain0.pt"
        retrain1_model_path = f"{args.model_dir}/retrain1.pt"
        retrain2_model_path = f"{args.model_dir}/retrain2.pt"
        retrain3_model_path = f"{args.model_dir}/retrain3.pt"
        retrain4_model_path = f"{args.model_dir}/retrain4.pt"
        retrain5_model_path = f"{args.model_dir}/retrain5.pt"
        retrain6_model_path = f"{args.model_dir}/retrain6.pt"
        retrain7_model_path = f"{args.model_dir}/retrain7.pt"
        retrain8_model_path = f"{args.model_dir}/retrain8.pt"
        retrain9_model_path = f"{args.model_dir}/retrain9.pt"

        retrain0_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain1_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain2_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain3_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain4_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain5_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain6_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain7_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain8_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        retrain9_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)

        utils.load_model_weights(retrain0_model, retrain0_model_path, device)
        utils.load_model_weights(retrain1_model, retrain1_model_path, device)
        utils.load_model_weights(retrain2_model, retrain2_model_path, device)
        utils.load_model_weights(retrain3_model, retrain3_model_path, device)
        utils.load_model_weights(retrain4_model, retrain4_model_path, device)
        utils.load_model_weights(retrain5_model, retrain5_model_path, device)
        utils.load_model_weights(retrain6_model, retrain6_model_path, device)
        utils.load_model_weights(retrain7_model, retrain7_model_path, device)
        utils.load_model_weights(retrain8_model, retrain8_model_path, device)
        utils.load_model_weights(retrain9_model, retrain9_model_path, device)

        logger.info("Computing mean representations shift vectors...")
        model_dict = {
            "original": ori_model,
            "retrain0": retrain0_model,
            "retrain1": retrain1_model,
            "retrain2": retrain2_model,
            "retrain3": retrain3_model,
            "retrain4": retrain4_model,
            "retrain5": retrain5_model,
            "retrain6": retrain6_model,
            "retrain7": retrain7_model,
            "retrain8": retrain8_model,
            "retrain9": retrain9_model
        }
        reps_dict = analyse.extract_representation_from_n_models(model_dict, unlearn_loader, device, "none")

    if "gradient" in args.exps or "concentration" in args.exps:
        logger.info("Computing concentration basis...")
        H_r, y_r = repr_metrics.get_representations(retain_loader, ori_model)   # (N_r, 512) at theta_o
        B = theory.concentration_basis(H_r)

        mean_ori = reps_dict["original"].mean(0)
        # Compute shifts
        dhs = []
        for i in range(10):
            mean_retrain = reps_dict[f"retrain{i}"].mean(0)
            shift_retrain = mean_retrain - mean_ori
            dhs.append(shift_retrain)

        null_dir = []
        null_index = []
        for i in range(10):
            for j in range(i+1, 10):
                null_dir.append(reps_dict[f"retrain{j}"].mean(0) - reps_dict[f"retrain{i}"].mean(0))
                null_index.append(f"retrain{i}_retrain{j}")

        W = ori_model.fc.weight.detach().cpu()   # (C, d)
        b = ori_model.fc.bias.detach().cpu()     # (C,)

        print(f"ori_model.fc.weight shape: {W.shape}")
        print(f"ori_model.fc.bias shape: {b.shape}")

        assert W.shape[0] == num_classes and W.shape[1] == H_r.shape[1]

        logits_r, _ = repr_metrics.get_logits(retain_loader, ori_model)
        logits_manual = H_r @ W.T + b

        diff = (logits_r - logits_manual).abs()
        print("max abs diff :", diff.max().item())
        print("mean abs diff:", diff.mean().item())
        print("allclose     :", torch.allclose(logits_r, logits_manual, atol=1e-4, rtol=1e-4))

        curv = theory.feature_loss_curvature(H_r, W, b, B["centered"]["eigvecs"], dhs=dhs, null_dir=null_dir)

    if "concentration" in args.exps:
        # spectra agree off the top: compare from rank 1 onward
        c, u = B["centered"]["eigvals"], B["uncentered"]["eigvals"]
        logger.info(f"top eigval  centered/uncentered: {c[0].item()} {u[0].item()}")
        logger.info(
            f"tail rel-diff (rank>=1): "
            f"{(c[1:] - u[1:]).abs().div(u[1:].clamp_min(1e-12)).max().item()}")

        # the mean should load almost entirely on the uncentered top eigenvector
        v1_u = B["uncentered"]["eigvecs"][:, 0]
        mean_hat = B["mean"] / B["mean"].norm()
        logger.info(f"cos(h_bar, v1_uncentered): {torch.dot(mean_hat, v1_u).abs().item()}")

        logger.info(f"centered   dust rel_neg: {B['centered']['neg_diag']['rel_neg']}")
        logger.info(f"uncentered dust rel_neg: {B['uncentered']['neg_diag']['rel_neg']}")

        # 1. WHERE is the mismatch? (predict: rank 1-3, decaying toward the deep tail)
        reldiff = (c[1:] - u[1:]).abs() / u[1:].clamp_min(1e-12)
        k = int(reldiff.argmax()) + 1
        print(f"max tail rel-diff {reldiff.max():.3f} at rank {k}")
        for r in range(12):
            print(f"  rank {r:>2}: c {c[r]:8.4f}  u {u[r]:8.4f}  rel {(abs(c[r]-u[r])/max(u[r].item(),1e-12)):.4f}")

        # 2. interlacing test: does c[k] match u[k+1]? (shift-by-one)
        shift1 = (c[:-1] - u[1:]).abs() / u[1:].clamp_min(1e-12)
        print("max rel-diff  c[k] vs u[k+1] (shift-by-one):", shift1.max().item())

        # 3. mean magnitude consistency: ||h_bar||^2 should equal trace(M) - trace(C)
        print("||h_bar||^2       =", B["mean"].pow(2).sum().item())
        print("trace(M)-trace(C) =", (u.sum() - c.sum()).item())

        logger.info("Computing concentration curves...")
        ## code for curvature overlay
        # dhs: (10, 512) forget-set mean-shift vectors; B from concentration_basis(H_r)
        c_evec = B["centered"]["eigvecs"]                   # (512, 512), columns, descending
        c_curves = theory.concentration_curves(dhs, c_evec, n_random=1, seed=0)
        """
        c_fig, ax = theory.plot_concentration(c_curves)
        c_fig.savefig(f"{output_path}centered_concentration_curve.png", dpi=150)

        u_evec = B["uncentered"]["eigvecs"]                   # (512, 512), columns, descending
        u_curves = theory.concentration_curves(dhs, u_evec, n_random=1, seed=0)

        u_fig, ax = theory.plot_concentration(u_curves)
        u_fig.savefig(f"{output_path}uncentered_concentration_curve.png", dpi=150)

        # --- summary statistic: fraction of shift mass in the low-variance half ---
        c_d = c_evec.shape[0]
        c_tail_frac = 1.0 - c_curves["shift"][:, c_d // 2 - 1]      # mass beyond the top c_d/2 directions
        c_rand_tail = 1.0 - c_curves["random"][:, c_d // 2 - 1]
        logger.info(f"centered shift  low-variance-half mass: {c_tail_frac.mean():.3f} ± {c_tail_frac.std():.3f}")
        logger.info(f"centered random low-variance-half mass: {c_rand_tail.mean():.3f}")

        logger.info(f"centered shift sq mass: {c_curves['shift squared mass'].squeeze(-1).tolist()}")
        logger.info(f"centered random sq mass: {c_curves['random squared mass'].squeeze(-1).tolist()}")

        u_d = u_evec.shape[0]
        u_tail_frac = 1.0 - u_curves["shift"][:, u_d // 2 - 1]      # mass beyond the top u_d/2 directions
        u_rand_tail = 1.0 - u_curves["random"][:, u_d // 2 - 1]
        logger.info(f"uncentered shift  low-variance-half mass: {u_tail_frac.mean():.3f} ± {u_tail_frac.std():.3f}")
        logger.info(f"uncentered random low-variance-half mass: {u_rand_tail.mean():.3f}")

        logger.info(f"uncentered shift sq mass: {u_curves['shift squared mass'].squeeze(-1).tolist()}")
        logger.info(f"uncentered random sq mass: {u_curves['random squared mass'].squeeze(-1).tolist()}")

        c_mm = theory.diagnose_concentration(dhs, B["centered"]["eigvals"], B["centered"]["eigvecs"])
        u_mm = theory.diagnose_concentration(dhs, B["uncentered"]["eigvals"], B["uncentered"]["eigvecs"])

        mass_mean, var_cum = theory.diagnose_by_variance(dhs, B["centered"]["eigvals"], B["centered"]["eigvecs"])
        mass_mean, var_cum = theory.diagnose_by_variance(dhs, B["uncentered"]["eigvals"], B["uncentered"]["eigvecs"])

        c_eval = B["centered"]["eigvals"]
        cv_fig, ax = theory.plot_concentration_by_variance(dhs, c_eval, c_evec)
        cv_fig.savefig(f"{output_path}centered_concentration_variance_curve.png", dpi=150)

        u_eval = B["uncentered"]["eigvals"]
        uv_fig, ax = theory.plot_concentration_by_variance(dhs, u_eval, u_evec)
        uv_fig.savefig(f"{output_path}uncentered_concentration_variance_curve.png", dpi=150)

        align = theory.resolve_concentration(dhs, B["centered"]["eigvals"], B["centered"]["eigvecs"])
        align = theory.resolve_concentration(dhs, B["uncentered"]["eigvals"], B["uncentered"]["eigvecs"])

        contrib = theory.settle_it(dhs, B["centered"]["eigvals"], B["centered"]["eigvecs"])
        contrib = theory.settle_it(dhs, B["uncentered"]["eigvals"], B["uncentered"]["eigvecs"])
        """
        ## actual concentration code
        #c_all_rows = theory.full_concentration_report(dhs, B["centered"]["eigvals"], B["centered"]["eigvecs"], f"{output_path}centered")
        #u_all_rows = theory.full_concentration_report(dhs, B["uncentered"]["eigvals"], B["uncentered"]["eigvecs"], f"{output_path}uncentered")

        # both curves on the cumulative-VARIANCE axis
        ev = B["centered"]["eigvals"]
        ev_frac = (ev / ev.sum())                 # per-rank variance fraction
        var_cum = ev_frac.cumsum(0)
        x = var_cum.cpu()                                  # cumulative variance (shared x)
        
        mass_cum_shift = c_curves["shift"].mean(0)
        mass_cum_random = c_curves["random"].mean(0)
        shift_c = mass_cum_shift.cpu()                     # shift cumulative mass (mean over 10)
        rand_c  = mass_cum_random.cpu()                    # random cumulative mass

        # gap = shift - random, on the variance axis. positive = more concentrated than chance here
        gap = shift_c - rand_c
        print("max |gap| at cum-variance =", x[gap.abs().argmax()].item(),
            " gap =", gap[gap.abs().argmax()].item())
        # sign of gap in the tail (variance > 0.74):
        tail = x > 0.74
        print("mean gap in tail (var>0.74):", gap[tail].mean().item(),
            " -> negative means tail is DEPLETED vs random")

        theory.overlay_feature_loss_curvature(c_curves, curv,
                                              out_dir=f"{output_path}loss_curvature/centered")

    if "finetune" in args.exps:
        retrain_model_path = f"{args.model_dir}/{args.retrain_model_name}.pt"
        retrain_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
        utils.load_model_weights(retrain_model, retrain_model_path, device)

        model_dict = {
            "original": ori_model,
            "retrain": retrain_model,
        }
        for i in range(1, 21):
            finetune_model_path = f"{args.model_dir}/{args.finetune_mode}finetune{i}.pt"
            finetune_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
            utils.load_model_weights(finetune_model, finetune_model_path, device)
            model_dict[i] = finetune_model

        reps_dict = analyse.extract_representation_from_n_models(model_dict, unlearn_loader, device, reduction="none")
        h_o = reps_dict["original"]
        h_rj = reps_dict["retrain"]
        h_ft = []
        for i in range(1, 21):
            h_ft.append(reps_dict[i])
        
        # h_o           : (N, D) original-model forget reps
        # h_rj          : (N, D) retrain j forget reps          (j = 0..9)
        # h_ft[t]       : (N, D) fine-tune epoch-t forget reps  (t = 1..20)

        results = theory.compute_parallel_residuals(h_o, h_rj, h_ft)   # h_ft = list len 20

        # stops if s_0 != -||Delta_j||
        try:
            p, s0, tgt, err = theory.sanity_check_s0(results, raise_on_fail=True)
        except Exception as e:                  # noqa
            logger.info(f"{args.retrain_model_name!s:>6}  ERROR: {e}")

        # if no error -> pass
        logger.info(f"{args.retrain_model_name!s:>6} {s0:16.6e} {tgt:16.6e} {err:10.2e}  {'PASS' if p else 'FAIL'}")

        # # bare weight-decay per-epoch factor:  exp(-lr * wd * S),  S = |D_r|/batch
        # import math
        # S = 45000 / 128
        # wd_rate = math.exp(-0.01 * 5e-4 * S)      # ~0.99824 for |D_r|=45k
        
        # or a single reference on its own:
        theory.plot_residual_decay(results["s_norm"], quantity="s",
                            title=f"run {args.finetune_mode}  {args.retrain_model_name}",
                            save_path=f"{output_path}fine_tune_on_retain/run_{args.finetune_mode}/s_{args.retrain_model_name}.png")
        
        theory.plot_residual_decay(results["Q_norm"], quantity="Q",
                            title=f"run {args.finetune_mode}  {args.retrain_model_name}",
                            save_path=f"{output_path}fine_tune_on_retain/run_{args.finetune_mode}/Q_{args.retrain_model_name}.png")

        # or single retrain (appends if the file exists):
        theory.save_residual_csv(results, retrain_name=args.retrain_model_name,
                          csv_path=f"{output_path}fine_tune_on_retain/run_{args.finetune_mode}/residuals.csv")

    if "gradient" in args.exps:
        res = theory.retain_gradient_alignment(
            H_r,          # (N_r, d) retain features at theta_o
            W,                # (C, d) final FC weight, logits = W @ h + b
            b,                # (C,) final FC bias
            y_r,         # (N_r,) integer retain labels  <-- REQUIRED
            B["centered"]["eigvecs"],          # (d, d) centered eigenvectors, columns, descending (Task C)
            dhs,              # (K, d) forget-set mean-shift vectors, or list of (d,)
            null_dir,
        )

        res["eig"]["curv"] = curv["c_eig"]
        res["random"]["curv"] = curv["c_rand"]
        res["shift"]["curv"] = curv["c_shift"]
        res["null"]["curv"] = curv["c_null"]

        theory.plot_all_shift_vs_spectrum(res, savedir=f"{output_path}gradient_curvature/centered")

        theory.save_alignment_csvs(res, null_index, output_dir=f"{output_path}gradient_curvature/centered")

    if "forget_shift" in args.exps:

        sample_dhs = []
        for i in range(10):
            sample_shift_retrain = reps_dict[f"retrain{i}"] - reps_dict["original"]
            sample_dhs.append(sample_shift_retrain)

        for i, sample_shift in enumerate(sample_dhs):
            pc_forget_shift = theory.concentration_basis(sample_shift)

            theory.explained_variance(pc_forget_shift, f"{output_path}forget_shift", csv_path=f"retrain{i}_evr.csv", plot_path=f"retrain{i}_evr.png")
            theory.mean_shift_alignment(pc_forget_shift, f"{output_path}forget_shift", csv_path=f"retrain{i}_align.csv")

            w_f = ori_model.fc.weight[args.unlearn_class].detach().cpu()
            pc_forget_curves = theory.concentration_curves([w_f], pc_forget_shift["centered"]["eigvecs"], n_random=1, seed=0)
            theory.plot_concentration_mass(pc_forget_curves, pc_forget_shift["centered"]["eigvals"], csv_path=f"{output_path}forget_shift/retrain{i}_concentration.csv", plot_path=f"{output_path}forget_shift/retrain{i}_concentration.png")

    metrics_dict = {}

    logger.info("Saving computed metrics...")
    with open(OUTPUT_METRICS_FILE, 'w') as f:
        yaml.safe_dump(metrics_dict, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    main(args)