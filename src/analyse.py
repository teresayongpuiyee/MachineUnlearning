import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src import repr_metrics

@torch.no_grad()
def extract_representation_from_n_models(model_dict, dataloader, device, reduction="mean"):
    rep_dict = dict()
    
    for model_key, model in model_dict.items():
        model.eval()
        if reduction == "mean":
            rep_dict[model_key] = None
        else:
            rep_dict[model_key] = []
    
    total_count = 0

    for x, _ in tqdm(dataloader):
        x = x.to(device, non_blocking=True)
        total_count += x.size(0)

        # Extract features for all three models
        for model_key, model in model_dict.items():
            h = model.feature_extractor(x)
            if reduction == "mean":
                if rep_dict[model_key] is None:
                    rep_dict[model_key] = torch.zeros(h.size(1), device=device)
                rep_dict[model_key] += h.sum(dim=0)
            else:
                rep_dict[model_key].append(h)

    if reduction == "mean":
        # Calculate means
        for model_key in rep_dict:
            rep_dict[model_key] /= total_count
    else:
        # Concatenate all representations
        for model_key in rep_dict:
            rep_dict[model_key] = torch.cat(rep_dict[model_key], dim=0)

    return rep_dict

def compute_per_sample_rep_shift_similarity(reps_ori, reps_retrain, reps_unlearn):
    shift_retrain = reps_retrain - reps_ori
    shift_unlearn = reps_unlearn - reps_ori

    # shape: [n] — one cosine similarity per sample pair
    per_sample_cos_sim = F.cosine_similarity(shift_retrain, shift_unlearn, dim=1)

    # scalar average
    mean_cos_sim = per_sample_cos_sim.mean().item()

    trans_cka = repr_metrics.linear_cka(shift_retrain, shift_unlearn)

    return mean_cos_sim, trans_cka

def compute_rep_shift_alignment(ori_model, retrain_model, unlearned_model, dataloader, device, unlearn_method, output_path, dataset_name):
    # Single pass over the data for mean representation extraction
    model_dict = {
        "original": ori_model,
        "retrain": retrain_model,
        "unlearn": unlearned_model
    }
    reps_dict = extract_representation_from_n_models(model_dict, dataloader, device, reduction="none")
    reps_ori = reps_dict["original"]
    reps_retrain = reps_dict["retrain"]
    reps_unlearn = reps_dict["unlearn"]

    mean_ori = reps_ori.mean(dim=0)
    mean_retrain = reps_retrain.mean(dim=0)
    mean_unlearn = reps_unlearn.mean(dim=0)

    mean_cos_sim, trans_cka = compute_per_sample_rep_shift_similarity(reps_ori, reps_retrain, reps_unlearn)

    #visualize_rep_shifts(mean_ori, mean_retrain, mean_unlearn, unlearn_method=unlearn_method, output_path=output_path, dataset_name=dataset_name)

    # Compute shifts
    shift_retrain = mean_retrain - mean_ori
    shift_unlearn = mean_unlearn - mean_ori
    
    # Compute magnitude
    mag_shift_retrain = torch.norm(shift_retrain, p=2).item()
    mag_shift_unlearn = torch.norm(shift_unlearn, p=2).item()
    
    # Directional alignment
    shift_cos_sim = F.cosine_similarity(shift_retrain.unsqueeze(0), shift_unlearn.unsqueeze(0)).item()

    # Relative magnitude (closer to 1.0 is better)
    mag_shift_ratio = mag_shift_unlearn / (mag_shift_retrain + 1e-9)

    # Breakdown metrics
    retrain_cos_sim = F.cosine_similarity(mean_retrain.unsqueeze(0), mean_ori.unsqueeze(0)).item()
    unlearn_cos_sim = F.cosine_similarity(mean_unlearn.unsqueeze(0), mean_ori.unsqueeze(0)).item()
    mag_retrain = torch.norm(mean_retrain, p=2).item()
    mag_unlearn = torch.norm(mean_unlearn, p=2).item()
    mag_ori = torch.norm(mean_ori, p=2).item()

    mag_retrain_ratio = mag_retrain / (mag_ori + 1e-9)
    mag_unlearn_ratio = mag_unlearn / (mag_ori + 1e-9)

    breakdown_metrics = {
        "retrain_cos_sim": round(retrain_cos_sim, 4),
        "unlearn_cos_sim": round(unlearn_cos_sim, 4),
        "mag_retrain": round(mag_retrain, 4),
        "mag_unlearn": round(mag_unlearn, 4),
        "mag_ori": round(mag_ori, 4),
        "mag_retrain_ratio": round(mag_retrain_ratio, 4),
        "mag_unlearn_ratio": round(mag_unlearn_ratio, 4),
        "mag_shift_retrain": round(mag_shift_retrain, 4),
        "mag_shift_unlearn": round(mag_shift_unlearn, 4)
    }

    mean_reps = {
        "mean_ori": mean_ori,
        "mean_retrain": mean_retrain,
        "mean_unlearn": mean_unlearn
    }

    return (
        breakdown_metrics, 
        round(shift_cos_sim, 4), 
        round(mag_shift_ratio, 4), 
        round(mean_cos_sim, 4),
        trans_cka,
        mean_reps
    )

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
    return round(h_mean, 4)

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

    return round(unlearn_ret_unl_cos_sim,4), round(retrain_ret_unl_cos_sim,4)

def visualize_rep_shifts(mean_ori, mean_retrain, mean_unlearn, labels=None, 
                         unlearn_method="", output_path=None, dataset_name=""):
    """
    Visualize the mean representations and their shifts in 2D using PCA.
    
    Args:
        mean_ori: Tensor, mean representation from original model (D,)
        mean_retrain: Tensor, mean representation from retrain model (D,)
        mean_unlearn: Tensor, mean representation from unlearned model (D,)
        labels: Optional list of strings for labeling points (default: ["Original", "Retrain", "Unlearned"])
        title: Plot title
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
        save_path = f"{output_path}rep_shift_{unlearn_method}_{dataset_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

def project_representations(
    representations, ori_model, retrain_model, dataloader, device, projection=""
):
    target_device = representations.device  # should be cpu

    model_dict = {
        "original": ori_model,
        "retrain": retrain_model,
    }
    mean_reps_dict = extract_representation_from_n_models(model_dict, dataloader, device, reduction="mean")
    
    # Move mean reps to CPU to match representations
    mean_ori = mean_reps_dict["original"].to(target_device)
    mean_retrain = mean_reps_dict["retrain"].to(target_device)

    # Compute shift direction
    shift_retrain = mean_retrain - mean_ori
    shift_retrain_norm = torch.norm(shift_retrain)
    
    direction = shift_retrain / (shift_retrain_norm + 1e-8)  # Avoid division by zero (D,)

    # scalar projection onto direction
    scalar_proj = torch.matmul(representations, direction)  # (N,)

    # parallel component
    parallel = scalar_proj.unsqueeze(1) * direction.unsqueeze(0)  # (N, D)

    if "orthogonal" in projection:
        # Project representations orthogonally to the shift direction
        orthogonal = representations - parallel  # (N, D)
        return orthogonal
    elif "parallel" in projection:
        # Project representations parallel to the shift direction
        return parallel
    else:
        # Return original representations if neither orthogonal nor parallel projection is requested
        return representations