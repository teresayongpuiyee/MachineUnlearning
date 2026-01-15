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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from typing import Tuple, Optional

# t-SNE visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_representations(
    loader: DataLoader,
    model: torch.nn.Module,
):
    model.eval()
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

# Rep-MIA without balance and normalize features
def basic_rep_mia(
    retain_reps: torch.tensor,
    forget_reps: torch.tensor,
    test_reps: torch.tensor,
) -> float:
    # Prepare data for attack: retain (member, label=1), test (non-member, label=0)
    X = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf.fit(X, y)

    train_acc = clf.score(X, y) * 100
    train_preds = clf.predict(X)
    train_f1 = f1_score(y, train_preds, average="macro") * 100

    metrics_dict = {
        "train_acc": round(float(train_acc), 4),
        "train_f1": round(float(train_f1), 4),
    }

    # Attack on forget set (should be members)
    forget_pred = clf.predict(forget_reps.numpy())
    asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
    return metrics_dict, round(float(asr), 4)

# Rep-MIA with balance and normalize features
def badt_rep_mia(
    retain_reps: torch.tensor,
    forget_reps: torch.tensor,
    test_reps: torch.tensor,
    retain_labels: torch.tensor,
) -> float:
    # Subsampling of retain data
    target_size = test_reps.shape[0]

    indices = np.arange(len(retain_reps))
    _, sampled_indices = train_test_split(
        indices,
        test_size=target_size,
        stratify=retain_labels.numpy(),
        random_state=42
    )
    retain_reps = retain_reps[sampled_indices]

    # Prepare data for attack: retain (member, label=1), test (non-member, label=0)
    X_full = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y_full = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    X, X_test, y, y_test = train_test_split(
        X_full, 
        y_full, 
        test_size=0.2,
        stratify=y_full,
        random_state=42
    )

    # Feature normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    forget_X = scaler.transform(forget_reps.numpy())

    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf.fit(X, y)

    train_acc = clf.score(X, y) * 100
    train_preds = clf.predict(X)
    train_f1 = f1_score(y, train_preds, average="macro") * 100

    test_acc = clf.score(X_test, y_test) * 100
    test_preds = clf.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro") * 100

    metrics_dict = {
        "train_acc": round(float(train_acc), 4),
        "train_f1": round(float(train_f1), 4),
        "test_acc": round(float(test_acc), 4),
        "test_f1": round(float(test_f1), 4),
    }

    # Attack on forget set (should be members)
    forget_pred = clf.predict(forget_X)
    asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
    return metrics_dict, round(float(asr), 4)

# Representation-level Membership Inference Attack (MIA) using five-fold attack and linear regressor
# based on POUR: https://arxiv.org/abs/2511.19339 
def pour_rmia(
    train_reps: torch.tensor,
    test_reps: torch.tensor,
    train_labels: torch.tensor,
    test_labels: torch.tensor,
    unlearn_class: int
) -> Tuple[dict, Optional[float]]:
    """
    Representation-level membership-inference attack success rate on forget set. Perform a five-fold attack
    using a linear regressor on the representation between the train and test sets
    Args:
        train_reps: Representations for the train set (member)
        test_reps: Representations for the test set (non-member)
        train_labels: Labels for the train set
        test_labels: Labels for the test set
        unlearn_class: The class label that was unlearned (forgotten)
    Returns:
        Attack model metrics (dict)
        Attack success rate (float or None): Percent of forget samples classified as train/member)
    """
    # Subsampling to balance Member (1) and Non-Member (0) classes
    target_size = test_reps.shape[0]
    if len(train_reps) > target_size:
        indices = np.arange(len(train_reps))
        # Stratify by labels to ensure we don't lose the unlearn_class during sampling
        _, sampled_indices = train_test_split(
            indices,
            test_size=target_size,
            stratify=train_labels.numpy(),
            random_state=42
        )
        train_reps = train_reps[sampled_indices]
        train_labels = train_labels[sampled_indices]
    
    # Prepare data for attack
    X_full = torch.cat([train_reps, test_reps], dim=0).numpy()
    X_labels = np.concatenate([train_labels.numpy(), test_labels.numpy()])
    y_full = np.concatenate([np.ones(len(train_reps)), np.zeros(len(test_reps))])
    
    strat_key = np.array([f"{lbl}_{mem}" for lbl, mem in zip(X_labels, y_full)])

    # Five-fold cross-validation attack
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {"train_acc": [], "train_f1": [], "test_acc": [], "test_f1": [], "forget_asr": []}

    for train_idx, test_idx in kf.split(X_full, strat_key):
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_test, y_test, X_test_labels = X_full[test_idx], y_full[test_idx], X_labels[test_idx]

        # Feature normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y_train)

        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        
        results["train_acc"].append(clf.score(X_train, y_train))
        results["train_f1"].append(f1_score(y_train, train_preds, average="macro"))
        results["test_acc"].append(clf.score(X_test, y_test))
        results["test_f1"].append(f1_score(y_test, test_preds, average="macro"))

        # MIA logic: How many 'forgotten' samples are predicted as Members (label 1)?
        mask = (X_test_labels == unlearn_class) & (y_test == 1)
        if mask.any():
            forget_preds = clf.predict(X_test[mask])
            results["forget_asr"].append(forget_preds.mean())

    metrics_dict = {
        "train_acc": round(float(np.mean(results["train_acc"])* 100), 4),
        "train_f1": round(float(np.mean(results["train_f1"])* 100), 4),
        "test_acc": round(float(np.mean(results["test_acc"])* 100), 4),
        "test_f1": round(float(np.mean(results["test_f1"])* 100), 4),
    }

    forget_asr =None
    if results["forget_asr"]:
        forget_asr = round(float(np.mean(results["forget_asr"]) * 100), 4)

    return metrics_dict, forget_asr


# MIA in Representation Space based on SURE: https://openreview.net/forum?id=KzSGJy1PIf
def sure_miars(
    train_reps: torch.tensor,
    test_reps: torch.tensor,
    train_labels: torch.tensor,
    test_labels: torch.tensor,
    unlearn_class: int,
    n_neighbors: int = 5,
) -> Tuple[dict, Optional[float]]:
    """
    Trains a KNN classifier to distinguish between train and test samples based on their representations,
    then applies the trained KNN to classify the forget samples and calculates the attack success rate (ASR).
    Args:
        train_reps: Representations for the train set (member)
        test_reps: Representations for the test set (non-member)
        train_labels: Labels for the train set
        test_labels: Labels for the test set
        unlearn_class: The class label that was unlearned (forgotten)
        n_neighbors: Number of neighbors for KNN
    Returns:
        Attack model metrics (dict)
        Attack success rate (float or None): Percent of forget samples classified as train/member)
    """
    # Subsampling to balance Member (1) and Non-Member (0) classes
    target_size = test_reps.shape[0]
    if len(train_reps) > target_size:
        indices = np.arange(len(train_reps))
        # Stratify by labels to ensure we don't lose the unlearn_class during sampling
        _, sampled_indices = train_test_split(
            indices,
            test_size=target_size,
            stratify=train_labels.numpy(),
            random_state=42
        )
        train_reps = train_reps[sampled_indices]
        train_labels = train_labels[sampled_indices]

    # Prepare data for attack
    X_full = torch.cat([train_reps, test_reps], dim=0).numpy()
    X_labels = np.concatenate([train_labels.numpy(), test_labels.numpy()])
    y_full = np.concatenate([np.ones(len(train_reps)), np.zeros(len(test_reps))])

    strat_key = np.array([f"{lbl}_{mem}" for lbl, mem in zip(X_labels, y_full)])

    X_train, X_test, y_train, y_test, _, X_test_labels = train_test_split(
        X_full,
        y_full,
        X_labels,
        test_size=0.2,
        stratify=strat_key,   # Stratify using the combined key
        random_state=42
    )

    # Feature normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    train_acc = knn.score(X_train, y_train)
    train_preds = knn.predict(X_train)
    train_f1 = f1_score(y_train, train_preds, average="macro")

    test_acc = knn.score(X_test, y_test)
    test_preds = knn.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    # MIA logic: How many 'forgotten' samples are predicted as Members (label 1)?
    mask = (X_test_labels == unlearn_class) & (y_test == 1)
    if mask.any():
        forget_preds = knn.predict(X_test[mask])
        forget_asr = round(float(forget_preds.mean() * 100), 4)
    else:
        forget_asr = None

    metrics_dict = {
        "train_acc": round(float(train_acc * 100), 4),
        "train_f1": round(float(train_f1 * 100), 4),
        "test_acc": round(float(test_acc * 100), 4),
        "test_f1": round(float(test_f1 * 100), 4),
    }

    return metrics_dict, forget_asr

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

    Reference: ESC - https://github.com/KU-VGI/ESC/blob/main/evaluation.py
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
    reps: torch.tensor,
    all_labels: torch.tensor,
    unlearn_method: str,
    exp_name: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    max_samples: int = 10000,
):
    """
    Visualize representations using t-SNE.
    Args:
        reps: Torch tensor of shape (N, D) with representations
        all_labels: Torch tensor of shape (N,) with labels
        unlearn_method: Name for title / filename
        exp_name: Folder name to save visualization
        perplexity: t-SNE perplexity
        n_iter: Number of t-SNE iterations
        max_samples: Max number of points to visualize (subsampling)
    """
    reps = reps.numpy()
    all_labels = all_labels.numpy()

    # Only subsample if dataset is large
    if len(reps) > max_samples:
        reps, _, all_labels, _ = train_test_split(
            reps,
            all_labels,
            train_size=max_samples,
            stratify=all_labels,  # preserves class ratios
            random_state=42
        )

    # Standardize
    reps = StandardScaler().fit_transform(reps)

    # PCA for speed (retain 50 components or less if input dim < 50)
    if reps.shape[1] > 50:
        reps = PCA(n_components=50, random_state=42).fit_transform(reps)

    # Fast t-SNE using openTSNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        metric="euclidean",
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )
    reps_2d = tsne.fit(reps)

    # Ensure consistent color mapping: map each label to a specific color
    unique_labels = np.unique(all_labels)
    cmap_name = 'tab10' if len(unique_labels) <= 10 else 'tab20'
    base_cmap = plt.get_cmap(cmap_name)
    color_list = [base_cmap(i % base_cmap.N) for i in range(len(unique_labels))]
    label_to_color_idx = {label: idx for idx, label in enumerate(unique_labels)}
    color_indices = np.array([label_to_color_idx[label] for label in all_labels])
    custom_cmap = mcolors.ListedColormap(color_list)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=color_indices, cmap=custom_cmap, alpha=0.7)
    plt.title(f"t-SNE Visualization - {unlearn_method}")
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)), label='Label')
    cbar.ax.set_yticklabels([str(l) for l in unique_labels])
    plt.tight_layout()

    # Save figure
    save_path = "/".join([".", exp_name, "visualize"])
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(save_path + f"/tsne_{unlearn_method}.png")
    plt.show()