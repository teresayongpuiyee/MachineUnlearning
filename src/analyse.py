import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@torch.no_grad()
def extract_mean_representation_from_n_models(model_dict, dataloader, device):
    mean_dict = dict()
    
    for model_key, model in model_dict.items():
        model.eval()
        mean_dict[model_key] = None
    
    total_count = 0

    for x, _ in tqdm(dataloader):
        x = x.to(device, non_blocking=True)
        total_count += x.size(0)

        # Extract features for all three models
        for model_key, model in model_dict.items():
            # TODO: flag all_layer and index layer of interest as h
            h = model.feature_extractor(x)
            if mean_dict[model_key] is None:
                mean_dict[model_key] = torch.zeros(h.size(1), device=device)
            mean_dict[model_key] += h.sum(dim=0)

    # Calculate means
    for model_key in mean_dict:
        mean_dict[model_key] /= total_count

    return mean_dict

def compute_rep_shift_alignment(ori_model, retrain_model, unlearned_model, dataloader, device, unlearn_method, retrain_model_name, output_path, dataset_name, metrics):
    # Single pass over the data for mean representation extraction
    model_dict = {
        "original": ori_model,
        "retrain": retrain_model,
        "unlearn": unlearned_model
    }
    mean_reps_dict = extract_mean_representation_from_n_models(model_dict, dataloader, device)
    mean_ori = mean_reps_dict["original"]
    mean_retrain = mean_reps_dict["retrain"]
    mean_unlearn = mean_reps_dict["unlearn"]

    # Compute shifts
    shift_retrain = mean_retrain - mean_ori
    shift_unlearn = mean_unlearn - mean_ori

    if "visualize" in metrics:
        visualize_rep_shifts(mean_ori, mean_retrain, mean_unlearn, unlearn_method=f"{unlearn_method}_{retrain_model_name}", output_path=output_path, dataset_name=dataset_name)

    cosine = dict()
    mag_ratio = dict()

    # Breakdown metrics
    if "cosine" in metrics:
        retrain_cos_sim = F.cosine_similarity(mean_retrain.unsqueeze(0), mean_ori.unsqueeze(0)).item()
        unlearn_cos_sim = F.cosine_similarity(mean_unlearn.unsqueeze(0), mean_ori.unsqueeze(0)).item()

        # Directional alignment
        shift_cos_sim = F.cosine_similarity(shift_retrain.unsqueeze(0), shift_unlearn.unsqueeze(0)).item()

        cosine = {
            "retrain_cos_sim": retrain_cos_sim,
            "unlearn_cos_sim": unlearn_cos_sim,
            "shift_cos_sim": shift_cos_sim
        }
    
    if "magnitude" in metrics:
        mag_retrain = torch.norm(mean_retrain, p=2).item()
        mag_unlearn = torch.norm(mean_unlearn, p=2).item()
        mag_ori = torch.norm(mean_ori, p=2).item()

        mag_retrain_ratio = mag_retrain / (mag_ori + 1e-9)
        mag_unlearn_ratio = mag_unlearn / (mag_ori + 1e-9)

        # Compute magnitude
        mag_shift_retrain = torch.norm(shift_retrain, p=2).item()
        mag_shift_unlearn = torch.norm(shift_unlearn, p=2).item()

        # Relative magnitude (closer to 1.0 is better)
        mag_shift_ratio = mag_shift_unlearn / (mag_shift_retrain + 1e-9)

        mag_ratio = {
            "mag_retrain": mag_retrain,
            "mag_unlearn": mag_unlearn,
            "mag_ori": mag_ori,
            "mag_retrain_ratio": mag_retrain_ratio,
            "mag_unlearn_ratio": mag_unlearn_ratio,
            "mag_shift_retrain": mag_shift_retrain,
            "mag_shift_unlearn": mag_shift_unlearn,
            "mag_shift_ratio": mag_shift_ratio
        }

    mean_reps = {
        "mean_ori": mean_ori,
        "mean_retrain": mean_retrain,
        "mean_unlearn": mean_unlearn
    }

    breakdown_metrics = {
        "cosine_similarity": cosine,
        "magnitude_ratio": mag_ratio
    }

    return breakdown_metrics, mean_reps

def calculate_harmonic_mean(sim_retain, sim_unlearn):
    """
    Computes the harmonic mean of similarities. 
    Treats negative similarities (moving in the wrong direction) as 0.
    """
    # Clamp to [0, 1] range. 
    # Negative similarity is a failure to align, so it becomes 0.
    a = max(sim_retain, 0.0)
    b = max(sim_unlearn, 0.0)
    
    # Handle the zero case to avoid DivisionByZero
    if (a + b) <= 0:
        return 0.0
    
    # Standard harmonic mean formula
    h_mean = (2 * a * b) / (a + b)
    return h_mean

def compute_forget_retain_cosine_similarity(mean_reps_retain, mean_reps_unlearn):
    mean_ori_retain = mean_reps_retain["mean_ori"]
    mean_retrain_retain = mean_reps_retain["mean_retrain"]
    mean_unlearn_retain = mean_reps_retain["mean_unlearn"]

    mean_ori_unlearn = mean_reps_unlearn["mean_ori"]
    mean_retrain_unlearn = mean_reps_unlearn["mean_retrain"]
    mean_unlearn_unlearn = mean_reps_unlearn["mean_unlearn"]

    unlearn_retain_shift = mean_unlearn_retain - mean_ori_retain
    unlearn_unlearn_shift = mean_unlearn_unlearn - mean_ori_unlearn

    retrain_retain_shift = mean_retrain_retain - mean_ori_retain
    retrain_unlearn_shift = mean_retrain_unlearn - mean_ori_unlearn

    unlearn_ret_unl_cos_sim = F.cosine_similarity(unlearn_retain_shift.unsqueeze(0), unlearn_unlearn_shift.unsqueeze(0)).item()
    retrain_ret_unl_cos_sim = F.cosine_similarity(retrain_retain_shift.unsqueeze(0), retrain_unlearn_shift.unsqueeze(0)).item()

    return unlearn_ret_unl_cos_sim, retrain_ret_unl_cos_sim

def visualize_rep_shifts(mean_ori, mean_retrain, mean_unlearn, labels=None, 
                         unlearn_method="", output_path=None, dataset_name=""):
    """
    Visualize the mean representations and their shifts in 2D using PCA.
    
    Args:
        mean_ori: Tensor, mean representation from original model (D,)
        mean_retrain: Tensor, mean representation from retrain model (D,)
        mean_unlearn: Tensor, mean representation from unlearned model (D,)
        labels: Optional list of strings for labeling points (default: ["Original", "Retrain", "Unlearned"])
        unlearn_method: Unlearning method name for title purposes
        output_path: Optional file path to save the plot (e.g., "rep_shift.png")
        dataset_name: Name of the dataset for title purposes
    """
    # Stack embeddings
    reps = torch.stack([mean_ori, mean_retrain, mean_unlearn]).cpu().numpy()
    
    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    reps_2d = pca.fit_transform(reps)
    
    # Default labels
    if labels is None:
        labels = ["Original", "Retrain", "Unlearned"]
    
    # Plot points
    plt.figure(figsize=(6,6))
    plt.scatter(reps_2d[:,0], reps_2d[:,1], color=['blue', 'green', 'red'], s=100)
    
    # Annotate points
    for i, label in enumerate(labels):
        plt.text(reps_2d[i,0]+0.02, reps_2d[i,1]+0.02, label, fontsize=12)
    
    # Draw arrows for shifts
    plt.arrow(reps_2d[0,0], reps_2d[0,1],
              reps_2d[1,0]-reps_2d[0,0], reps_2d[1,1]-reps_2d[0,1],
              color='green', width=0.005, head_width=0.05, length_includes_head=True, label='Original→Retrain')
    
    plt.arrow(reps_2d[0,0], reps_2d[0,1],
              reps_2d[2,0]-reps_2d[0,0], reps_2d[2,1]-reps_2d[0,1],
              color='red', width=0.005, head_width=0.05, length_includes_head=True, label='Original→Unlearned')
    
    plt.title(f"Representation Shifts - {unlearn_method} ({dataset_name} set)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.axis('equal')
    
    # Save if path provided
    if output_path is not None:
        output_path = output_path + f"visualize"
        os.makedirs(output_path, exist_ok=True)
        save_path = f"{output_path}/rep_shift_{unlearn_method}_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

def project_representations(
    representations, ori_model, retrain_model, dataloader, device, projection="", random=False, seed=0
):
    target_device = representations.device  # should be cpu

    if not random:
        model_dict = {
            "original": ori_model,
            "retrain": retrain_model,
        }
        mean_reps_dict = extract_mean_representation_from_n_models(model_dict, dataloader, device)
        
        # Move mean reps to CPU to match representations
        mean_ori = mean_reps_dict["original"].to(target_device)
        mean_retrain = mean_reps_dict["retrain"].to(target_device)

        # Compute shift direction
        shift_retrain = mean_retrain - mean_ori
        shift_retrain_norm = torch.norm(shift_retrain)
        
        direction = shift_retrain / (shift_retrain_norm + 1e-8)  # Avoid division by zero (D,)

        norm_out = shift_retrain_norm.item()
    elif random:
        # Random baseline direction, reproducible from `seed` alone (independent of
        # how much global RNG earlier steps consumed). Using the SAME `seed` for
        # projection="parallel" and projection="orthogonal" yields matched 1-D and
        # (d-1)-D components from one direction; loop over seeds to build the null.
        generator = torch.Generator(device=target_device).manual_seed(seed)
 
        # Isotropic: uniform on the unit sphere in R^D.
        rand_vec = torch.randn(
            representations.shape[1], generator=generator,
            dtype=representations.dtype, device=target_device,
        )
 
        direction = rand_vec / (torch.norm(rand_vec) + 1e-8)  # (D,)
 
        # The second return value is the RETRAINING shift magnitude, which is
        # undefined for a random direction. Return None so it can't be silently
        # consumed as if it were a real shift norm.
        norm_out = None

    # scalar projection onto direction
    scalar_proj = torch.matmul(representations, direction)  # (N,)

    # parallel component
    parallel = scalar_proj.unsqueeze(1) * direction.unsqueeze(0)  # (N, D)

    if "orthogonal" in projection:
        # Project representations orthogonally to the shift direction
        orthogonal = representations - parallel  # (N, D)
        return orthogonal, norm_out
    elif "parallel" in projection:
        # Project representations parallel to the shift direction
        return parallel, norm_out
    else:
        # Return original representations if neither orthogonal nor parallel projection is requested
        return representations, norm_out


def summarize_against_null(observed, null_values):
    """
    Report the observed (retraining-direction) metric relative to the
    random-direction null built from the seed loop.

    Returns the null's 95% two-sided percentile band [p2.5, p97.5] for
    presentation, plus add-one-corrected one-sided p-values. Report the
    one-sided p that matches the pre-registered direction of the effect.
    """
    observed = float(observed)
    null = torch.as_tensor(list(null_values), dtype=torch.float64)
    n = null.numel()

    q = torch.quantile(null, torch.tensor([0.025, 0.975], dtype=torch.float64))
    p025, p975 = q[0].item(), q[1].item()

    return {
        "observed": observed,
        "n_random": n,
        "null_mean": null.mean().item(),
        "null_std": null.std(unbiased=True).item(),
        "null_p2.5": p025,
        "null_p97.5": p975,
        "percentile": (null < observed).double().mean().item() * 100.0,
        # add-one corrected one-sided p-values (Phipson & Smyth, 2010):
        "p_obs_greater_than_null": (1 + (null >= observed).sum().item()) / (1 + n),  # obs in UPPER tail
        "p_obs_less_than_null":    (1 + (null <= observed).sum().item()) / (1 + n),  # obs in LOWER tail
        "outside_95_band": (observed < p025) or (observed > p975),
    }