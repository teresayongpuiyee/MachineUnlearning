import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from sklearn.model_selection import KFold

# t-SNE visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_representations(
    loader: DataLoader,
    model: torch.nn.Module,
):
    loader = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False
    )
    reps = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            feat = model.feature_extractor(data)
            feat = feat.view(feat.size(0), -1)
            reps.append(feat.detach().cpu())
            all_labels.append(target.cpu())
    return torch.cat(reps, dim=0), torch.cat(all_labels, dim=0)

# Representation-level Membership Inference Attack (MIA) using five-fold attack and linear regressor
# based on https://arxiv.org/abs/2511.19339 
def representation_level_mia(
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    test_loader: DataLoader,
    model: torch.nn.Module,
) -> float:
    """
    Perform a five-fold membership inference attack on the forget set using representations from a specified layer.
    Args:
        retain_loader: DataLoader for the retain (train) set
        forget_loader: DataLoader for the forget set (attack target)
        test_loader: DataLoader for the test set (non-member)
        model: The model to extract representations from
    Returns:
        Attack success rate (float, percent)

    Example usage:
        >>> from src import metrics
        >>> asr = metrics.representation_level_mia(
        ...     retain_loader=retain_loader,
        ...     forget_loader=unlearn_loader,
        ...     test_loader=test_loader,
        ...     model=unlearned_model,
        ... )
        >>> print(f"Representation-level MIA ASR: {asr}%")
    """
    model.eval()

    # Get representations for retain, forget, and test sets
    retain_reps, _ = get_representations(retain_loader, model)
    forget_reps, _ = get_representations(forget_loader, model)
    test_reps, _ = get_representations(test_loader, model)
    
    # Prepare data for attack: retain (member, label=1), test (non-member, label=0)
    X = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    # Five-fold cross-validation attack
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    asr_list = []
    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y_train)
        # Attack on forget set (should be members)
        forget_pred = clf.predict(forget_reps.numpy())
        asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
        asr_list.append(asr)
    return round(np.mean(asr_list), 4)


# MIA in Representation Space based on https://openreview.net/forum?id=KzSGJy1PIf
def miars(
    retain_loader: DataLoader,
    test_loader: DataLoader,
    forget_loader: DataLoader,
    model: torch.nn.Module,
    n_neighbors: int = 5
) -> float:
    """
    Trains a KNN classifier to distinguish between train and test samples based on their representations,
    then applies the trained KNN to classify the forget samples and calculates the attack success rate (ASR).
    Args:
        retain_loader: DataLoader for the train set
        test_loader: DataLoader for the test set
        forget_loader: DataLoader for the forget set
        model: The model to extract representations from
        n_neighbors: Number of neighbors for KNN
    Returns:
        Attack success rate (float, percent of forget samples classified as train/member)
    """
    model.eval()
    # Get representations
    retain_reps, _ = get_representations(retain_loader, model)
    test_reps, _ = get_representations(test_loader, model)
    forget_reps, _ = get_representations(forget_loader, model)

    X = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    forget_pred = knn.predict(forget_reps.numpy())
    asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
    return round(asr, 4)

def linear_probing(
    train_loader: DataLoader,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    model: torch.nn.Module,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
) -> dict:
    """
    Trains a linear probe (head) on top of frozen model representations using SGD and cross-entropy,
    then evaluates accuracy on both the retain and forget sets.

    Reference: https://github.com/KU-VGI/ESC/blob/main/evaluation.py
    Args:
        train_loader: DataLoader for training the linear head
        retain_loader: DataLoader for the retain (remaining) set
        forget_loader: DataLoader for the forget set
        model: Model to extract representations
        num_classes: Number of output classes
        epochs: Number of training epochs
        lr: Learning rate for SGD
    Returns:
        Dictionary with accuracy on retain and forget sets
    """
    model.eval()
    device = next(model.parameters()).device
    # Setup linear head
    # Infer from feature_extractor output
    dummy = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        feat = model.feature_extractor(dummy)
        feat = feat.view(feat.size(0), -1)
    head = nn.Linear(feat.size(1), num_classes).to(device)
    nn.init.xavier_normal_(head.weight)
    nn.init.zeros_(head.bias)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True

    optimizer = optim.SGD(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train linear head
    for _ in range(epochs):
        head.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                feat = model.feature_extractor(x)
                feat = feat.view(feat.size(0), -1)
            logits = head(feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    def eval_accuracy(loader):
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                feat = model.feature_extractor(x)
                feat = feat.view(feat.size(0), -1)
                logits = head(feat)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total if total > 0 else 0.0

    retain_acc = eval_accuracy(retain_loader)
    forget_acc = eval_accuracy(forget_loader)

    return {
        "retain_accuracy": round(retain_acc, 4),
        "forget_accuracy": round(forget_acc, 4)
    }

# t-SNE visualization function
def visualize_tsne(
    loader: DataLoader,
    model: torch.nn.Module,
    unlearn_method: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    save_path: str = "./visualize"
):
    """
    Visualize representations using t-SNE.
    Args:
        loader: DataLoader for the data to visualize
        model: Model to extract representations
        unlearn_method: Unlearning method name for title
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        save_path: Saves the plot to this path
    """
    model.eval()
    
    reps, all_labels = get_representations(loader, model)
    reps = reps.numpy()
    all_labels = all_labels.numpy()

    reps = StandardScaler().fit_transform(reps)
    if reps.shape[1] > 50:
        reps = PCA(n_components=50, random_state=42).fit_transform(reps)

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reps_2d = tsne.fit_transform(reps)

    ## Plotting visualization
    # Ensure consistent color mapping: map each label to a specific color
    unique_labels = np.unique(all_labels)
    # Use tab10 or tab20 depending on number of classes
    cmap_name = 'tab10' if len(unique_labels) <= 10 else 'tab20'
    base_cmap = plt.get_cmap(cmap_name)
    # Build a ListedColormap with colors assigned in order of sorted unique_labels
    color_list = [base_cmap(i % base_cmap.N) for i in range(len(unique_labels))]
    label_to_color_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = np.array([label_to_color_idx[label] for label in all_labels])
    custom_cmap = mcolors.ListedColormap(color_list)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=color_indices, cmap=custom_cmap, alpha=0.7)
    plt.title("t-SNE Visualization - "+f"{unlearn_method}")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # Custom colorbar with correct label ticks
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Label')
    cbar.ax.set_yticklabels([str(l) for l in unique_labels])
    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + f"/tsne_{unlearn_method}.png")
    plt.show()