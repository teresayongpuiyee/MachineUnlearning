import os
import torch
import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from sklearn.model_selection import KFold

# t-SNE visualization
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_representations(
    loader: DataLoader,
    model: torch.nn.Module,
):
    loader = DataLoader(
        loader.dataset, batch_size=1, shuffle=False
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
    train_loader: DataLoader,
    test_loader: DataLoader,
    forget_loader: DataLoader,
    model: torch.nn.Module,
    n_neighbors: int = 5
) -> float:
    """
    Trains a KNN classifier to distinguish between train and test samples based on their representations,
    then applies the trained KNN to classify the forget samples and calculates the attack success rate (ASR).
    Args:
        train_loader: DataLoader for the train set
        test_loader: DataLoader for the test set
        forget_loader: DataLoader for the forget set
        model: The model to extract representations from
        n_neighbors: Number of neighbors for KNN
    Returns:
        Attack success rate (float, percent of forget samples classified as train/member)
    """
    model.eval()
    # Get representations
    train_reps, _ = get_representations(train_loader, model)
    test_reps, _ = get_representations(test_loader, model)
    forget_reps, _ = get_representations(forget_loader, model)

    X = torch.cat([train_reps, test_reps], dim=0).numpy()
    y = np.concatenate([np.ones(len(train_reps)), np.zeros(len(test_reps))])

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)
    forget_pred = knn.predict(forget_reps.numpy())
    asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
    return round(asr, 4)

def linear_probe_accuracy(
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    model: torch.nn.Module,
    solver: str = "lbfgs",
    max_iter: int = 1000,
) -> dict:
    """
    Trains a linear classifier (logistic regression) on the representations from the retain set,
    then evaluates accuracy on both the forget and retain sets.
    Args:
        retain_loader: DataLoader for the retain (remaining) set
        forget_loader: DataLoader for the forget set
        model: Model to extract representations
        solver: Solver for LogisticRegression
        max_iter: Max iterations for LogisticRegression
    Returns:
        Dictionary with accuracy on retain and forget sets
    """
    model.eval()
    # Get representations and labels
    retain_reps, retain_labels = get_representations(retain_loader, model)
    forget_reps, forget_labels = get_representations(forget_loader, model)

    X_train = retain_reps.numpy()
    y_train = retain_labels.numpy()
    X_forget = forget_reps.numpy()
    y_forget = forget_labels.numpy()

    clf = LogisticRegression(class_weight="balanced", solver=solver, max_iter=max_iter)
    clf.fit(X_train, y_train)

    retain_acc = clf.score(X_train, y_train) * 100
    forget_acc = clf.score(X_forget, y_forget) * 100

    return {
        "retain_accuracy": round(retain_acc, 4),
        "forget_accuracy": round(forget_acc, 4)
    }

# t-SNE visualization function
def visualize_tsne(
    loader: DataLoader,
    model: torch.nn.Module,
    title: str = "t-SNE Visualization",
    perplexity: int = 30,
    n_iter: int = 1000,
    save_path: str = None
):
    """
    Visualize representations using t-SNE.
    Args:
        loader: DataLoader for the data to visualize
        model: Model to extract representations
        title: Plot title
        perplexity: t-SNE perplexity
        n_iter: t-SNE iterations
        save_path: Optional, if provided, saves the plot to this path
    """
    model.eval()
    
    reps, all_labels = get_representations(loader, model)
    reps = reps.numpy()
    all_labels = all_labels.numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reps_2d = tsne.fit_transform(reps)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=all_labels, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='Label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()