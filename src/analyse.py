import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

@torch.no_grad()
def extract_mean_representation(model, dataloader, device):
    model.eval()
    total_sum = None
    total_count = 0

    for x, _ in dataloader:
        x = x.to(device)
        h = model.feature_extractor(x)   
        h = h.view(h.size(0), -1)  # (N, D)

        if total_sum is None:
            total_sum = torch.zeros(h.size(1), device=device)
        
        total_sum += h.sum(dim=0)
        total_count += h.size(0)

    return total_sum / total_count   # (D,)

def get_shift_metrics(model_ori, model_target, dataloader, device):
    """Computes the vector difference and its magnitude."""
    mean_ori = extract_mean_representation(model_ori, dataloader, device)
    mean_target = extract_mean_representation(model_target, dataloader, device)

    # compute representation differences
    shift_vector = mean_target - mean_ori
    magnitude = torch.norm(shift_vector, p=2).item()
    
    return shift_vector, magnitude

def compute_rep_shift_alignment(ori_model, retrain_model, unlearned_model, dataloader, device):
    # Get shifts relative to the original model
    shift_retrain, mag_retrain = get_shift_metrics(ori_model, retrain_model, dataloader, device)
    shift_unlearn, mag_unlearn = get_shift_metrics(ori_model, unlearned_model, dataloader, device)
    
    # Directional alignment
    shift_cos_sim = F.cosine_similarity(shift_retrain.unsqueeze(0), shift_unlearn.unsqueeze(0)).item()

    # Relative magnitude (closer to 1.0 is better)
    mag_ratio = mag_unlearn / (mag_retrain + 1e-9)

    return shift_cos_sim, mag_ratio

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