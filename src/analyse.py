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

def compute_rep_shift_alignment(ori_model, retrain_model, unlearned_model, dataloader, device):
    # Single pass over the data for mean representation extraction
    mean_ori, mean_retrain, mean_unlearn = extract_mean_representation_from_three_models(ori_model, retrain_model, unlearned_model, dataloader, device)

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

def plot_representation_shifts(mean_ori, mean_retrain, mean_unlearn, title="Feature Space Shift"):
    # Stack means into a single matrix for PCA: (3, D)
    combined = torch.stack([mean_ori, mean_retrain, mean_unlearn]).cpu().numpy()
    
    # Project to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    
    plt.figure(figsize=(8, 6))
    labels = ['Original', 'Retrained', 'Unlearned']
    colors = ['black', 'blue', 'red']
    
    # Plot points
    for i in range(3):
        plt.scatter(coords[i, 0], coords[i, 1], c=colors[i], label=labels[i], s=100, zorder=5)
    
    # Draw vectors (shifts) starting from Original
    origin_x, origin_y = coords[0]
    
    # Vector to Retrained
    plt.arrow(origin_x, origin_y, coords[1, 0] - origin_x, coords[1, 1] - origin_y, 
              head_width=0.05, color='blue', alpha=0.3, label='Ideal Shift')
    
    # Vector to Unlearned
    plt.arrow(origin_x, origin_y, coords[2, 0] - origin_x, coords[2, 1] - origin_y, 
              head_width=0.05, color='red', alpha=0.3, label='Actual Shift')

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()