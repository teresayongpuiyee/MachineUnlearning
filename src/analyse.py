import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@torch.no_grad()
def extract_mean_representation_from_three_models(ori_model, retrain_model, unlearned_model, dataloader, device):
    ori_model.eval()
    retrain_model.eval()
    unlearned_model.eval()
    
    sum_ori, sum_retrain, sum_unlearn = None, None, None
    total_count = 0

    for x, _ in tqdm(dataloader):
        x = x.to(device)
        total_count += x.size(0)

        # Extract features for all three models
        h_ori = ori_model.feature_extractor(x).view(x.size(0), -1)
        h_ret = retrain_model.feature_extractor(x).view(x.size(0), -1)
        h_unl = unlearned_model.feature_extractor(x).view(x.size(0), -1)

        if sum_ori is None:
            sum_ori = torch.zeros(h_ori.size(1), device=device)

        if sum_retrain is None:
            sum_retrain = torch.zeros(h_ret.size(1), device=device)

        if sum_unlearn is None:
            sum_unlearn = torch.zeros(h_unl.size(1), device=device)
        
        # Accumulate sums
        sum_ori += h_ori.sum(dim=0)
        sum_retrain += h_ret.sum(dim=0)
        sum_unlearn += h_unl.sum(dim=0)

    # Calculate means
    mean_ori = (sum_ori / total_count)
    mean_ret = (sum_retrain / total_count)
    mean_unl = (sum_unlearn / total_count)

    return mean_ori, mean_ret, mean_unl

def compute_rep_shift_alignment(ori_model, retrain_model, unlearned_model, dataloader, device, unlearn_method, output_path):
    # Single pass over the data for mean representation extraction
    mean_ori, mean_retrain, mean_unlearn = extract_mean_representation_from_three_models(ori_model, retrain_model, unlearned_model, dataloader, device)

    visualize_rep_shifts(mean_ori, mean_retrain, mean_unlearn, unlearn_method=unlearn_method, output_path=output_path)

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
        "mag_unlearn_ratio": round(mag_unlearn_ratio, 4)
    }

    return breakdown_metrics, round(shift_cos_sim, 4), round(mag_shift_ratio, 4)

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

def visualize_rep_shifts(mean_ori, mean_retrain, mean_unlearn, labels=None, 
                         unlearn_method="", output_path=None):
    """
    Visualize the mean representations and their shifts in 2D using PCA.
    
    Args:
        mean_ori: Tensor, mean representation from original model (D,)
        mean_retrain: Tensor, mean representation from retrain model (D,)
        mean_unlearn: Tensor, mean representation from unlearned model (D,)
        labels: Optional list of strings for labeling points (default: ["Original", "Retrain", "Unlearned"])
        title: Plot title
        output_path: Optional file path to save the plot (e.g., "rep_shift.png")
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
    
    plt.title(f"Representation Shifts - {unlearn_method}")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.axis('equal')
    
    # Save if path provided
    if output_path is not None:
        save_path = f"{output_path}rep_shift_{unlearn_method}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
