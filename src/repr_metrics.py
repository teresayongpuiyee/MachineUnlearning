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

def get_logits(
    loader: DataLoader,
    model: torch.nn.Module,
):
    model.eval()
    loader = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers, pin_memory=True, persistent_workers=True
    )
    logits = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = [tensor.to(next(model.parameters()).device, non_blocking=True) for tensor in batch]
            data, target = batch
            logit = model(data)
            logits.append(logit.detach().cpu())
            all_labels.append(target.cpu())
    return torch.cat(logits, dim=0), torch.cat(all_labels, dim=0)

def get_representations(
    loader: DataLoader,
    model: torch.nn.Module,
):
    model.eval()
    loader = DataLoader(
        loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers, pin_memory=True, persistent_workers=True
    )
    reps = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = [tensor.to(next(model.parameters()).device, non_blocking=True) for tensor in batch]
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
    test_labels: torch.tensor,
    unlearn_class: int,
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
    retain_labels = retain_labels[sampled_indices]

    # Prepare data for attack: retain (member, label=1), test (non-member, label=0)
    X_full = torch.cat([retain_reps, test_reps], dim=0).numpy()
    X_labels = np.concatenate([retain_labels.numpy(), test_labels.numpy()])
    y_full = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    strat_key = np.array([f"{lbl}_{mem}" for lbl, mem in zip(X_labels, y_full)])

    X, X_test, y, y_test, _, X_test_labels = train_test_split(
        X_full, 
        y_full, 
        X_labels,
        test_size=0.2,
        stratify=strat_key,
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

    # MIA logic: How many non member 'forgotten' samples are predicted as Members (label 1)?
    mask_nonmember = (X_test_labels == unlearn_class) & (y_test == 0)
    if mask_nonmember.any():
        forget_nonmember_preds = clf.predict(X_test[mask_nonmember])
        forget_nonmember_fpr = round(float(forget_nonmember_preds.mean() * 100), 4)
    else:
        forget_nonmember_fpr = None

    metrics_dict = {
        "train_acc": round(float(train_acc), 4),
        "train_f1": round(float(train_f1), 4),
        "test_acc": round(float(test_acc), 4),
        "test_f1": round(float(test_f1), 4),
        "forget_fpr": forget_nonmember_fpr
    }

    # Attack on forget set (should be members)
    forget_pred = clf.predict(forget_X)
    asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
    return metrics_dict, round(float(asr), 4)

def scrub_rep_mia(
    forget_reps: torch.tensor,
    test_reps: torch.tensor,
    test_labels: torch.tensor,
    unlearn_class: int,
) -> Tuple[dict, Optional[float]]:
    # Subsampling to balance Member (1) and Non-Member (0) classes
    target_size = forget_reps.shape[0]
    if len(test_reps) > target_size:
        indices = np.arange(len(test_reps))
        # Stratify by labels to ensure we don't lose the unlearn_class during sampling
        _, sampled_indices = train_test_split(
            indices,
            test_size=target_size,
            stratify=test_labels.numpy(),
            random_state=42
        )
        test_reps = test_reps[sampled_indices]
        test_labels = test_labels[sampled_indices]

    # Prepare data for attack
    forget_labels = torch.full_like(test_labels, fill_value=unlearn_class)
    X_full = torch.cat([forget_reps, test_reps], dim=0).numpy()
    X_labels = np.concatenate([forget_labels.numpy(), test_labels.numpy()])
    y_full = np.concatenate([np.ones(len(forget_reps)), np.zeros(len(test_reps))])

    strat_key = np.array([f"{lbl}_{mem}" for lbl, mem in zip(X_labels, y_full)])

    X_train, X_test, y_train, y_test, _, X_test_labels = train_test_split(
        X_full,
        y_full,
        X_labels,
        test_size=0.25,
        stratify=strat_key,   # Stratify using the combined key
        random_state=42
    )

    # Feature normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    train_preds = clf.predict(X_train)
    train_f1 = f1_score(y_train, train_preds, average="macro")

    test_acc = clf.score(X_test, y_test)
    test_preds = clf.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    # MIA logic: How many member 'forgotten' samples are predicted as Members (label 1)?
    mask = (X_test_labels == unlearn_class) & (y_test == 1)
    if mask.any():
        forget_preds = clf.predict(X_test[mask])
        forget_asr = round(float(forget_preds.mean() * 100), 4)
    else:
        forget_asr = None

    # MIA logic: How many non member 'forgotten' samples are predicted as Members (label 1)?
    mask_nonmember = (X_test_labels == unlearn_class) & (y_test == 0)
    if mask_nonmember.any():
        forget_nonmember_preds = clf.predict(X_test[mask_nonmember])
        forget_nonmember_fpr = round(float(forget_nonmember_preds.mean() * 100), 4)
    else:
        forget_nonmember_fpr = None

    metrics_dict = {
        "train_acc": round(float(train_acc * 100), 4),
        "train_f1": round(float(train_f1 * 100), 4),
        "test_acc": round(float(test_acc * 100), 4),
        "test_f1": round(float(test_f1 * 100), 4),
        "forget_fpr": forget_nonmember_fpr
    }

    return metrics_dict, forget_asr

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

    results = {"train_acc": [], "train_f1": [], "test_acc": [], "test_f1": [], "forget_fpr": [], "forget_asr": []}

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

        # MIA logic: How many non member 'forgotten' samples are predicted as Members (label 1)?
        mask_nonmember = (X_test_labels == unlearn_class) & (y_test == 0)
        if mask_nonmember.any():
            forget_nonmember_preds = clf.predict(X_test[mask_nonmember])
            results["forget_fpr"].append(forget_nonmember_preds.mean())

    metrics_dict = {
        "train_acc": round(float(np.mean(results["train_acc"])* 100), 4),
        "train_f1": round(float(np.mean(results["train_f1"])* 100), 4),
        "test_acc": round(float(np.mean(results["test_acc"])* 100), 4),
        "test_f1": round(float(np.mean(results["test_f1"])* 100), 4),
        "forget_fpr": round(float(np.mean(results["forget_fpr"])* 100), 4),
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

    # MIA logic: How many member 'forgotten' samples are predicted as Members (label 1)?
    mask = (X_test_labels == unlearn_class) & (y_test == 1)
    if mask.any():
        forget_preds = knn.predict(X_test[mask])
        forget_asr = round(float(forget_preds.mean() * 100), 4)
    else:
        forget_asr = None

    # MIA logic: How many non member 'forgotten' samples are predicted as Members (label 1)?
    mask_nonmember = (X_test_labels == unlearn_class) & (y_test == 0)
    if mask_nonmember.any():
        forget_nonmember_preds = knn.predict(X_test[mask_nonmember])
        forget_nonmember_fpr = round(float(forget_nonmember_preds.mean() * 100), 4)
    else:
        forget_nonmember_fpr = None

    metrics_dict = {
        "train_acc": round(float(train_acc * 100), 4),
        "train_f1": round(float(train_f1 * 100), 4),
        "test_acc": round(float(test_acc * 100), 4),
        "test_f1": round(float(test_f1 * 100), 4),
        "forget_fpr": forget_nonmember_fpr
    }

    return metrics_dict, forget_asr

def linear_probing(
    train_loader: DataLoader,
    retain_eval_loader: DataLoader,
    unlearn_eval_loader: DataLoader,
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
        retain_eval_loader: DataLoader for the retain (remaining) set
        unlearn_eval_loader: DataLoader for the forget set
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
    #optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        #scheduler.step()

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

    retain_acc = eval_accuracy(retain_eval_loader)
    forget_acc = eval_accuracy(unlearn_eval_loader)

    return {
        "retain_accuracy": round(retain_acc, 4),
        "forget_accuracy": round(forget_acc, 4)
    }

def binary_forget_probe(
    retain_eval_loader: DataLoader,   # representations of forget samples
    unlearn_eval_loader: DataLoader,   # representations of retain samples (subsample to balance)
    unlearned_model: torch.nn.Module
) -> dict:
    """
    Binary probe: can we linearly distinguish forget-class representations
    from non-forget representations? This is the core of experiment D.
    
    Labels: forget=1, retain=0
    Higher accuracy → forget information still encoded in representations.
    Should be ~50% for a perfectly unlearned model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import numpy as np

    forget_reps, _ = get_representations(unlearn_eval_loader, unlearned_model)
    retain_reps, _ = get_representations(retain_eval_loader, unlearned_model)

    # Balance classes
    n = min(len(forget_reps), len(retain_reps))
    X_forget = forget_reps[:n].cpu().numpy()
    X_retain = retain_reps[:n].cpu().numpy()
    
    X = np.concatenate([X_forget, X_retain], axis=0)
    y = np.array([1]*n + [0]*n)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    
    return {
        "binary_probe_acc_mean": round(float(scores.mean() * 100), 2),
        "binary_probe_acc_std": round(float(scores.std() * 100), 2),
    }

DISTINCT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#a9a9a9", "#000000", "#fffac8",
]

# t-SNE visualization function
def visualize_tsne(
    reps: torch.tensor,
    all_labels: torch.tensor,
    unlearn_method: str,
    save_path: str,
    perplexity: int = 30,
    n_iter: int = 1000,
    max_samples: int = 10000,
    tag: str = "",
):
    """
    Visualize representations using t-SNE.
    Args:
        reps: Torch tensor of shape (N, D) with representations
        all_labels: Torch tensor of shape (N,) with labels
        unlearn_method: Name for title / filename
        save_path: Folder path to save visualization
        perplexity: t-SNE perplexity
        n_iter: Number of t-SNE iterations
        max_samples: Max number of points to visualize (subsampling)
        tag: Additional tag for filename
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

    # Adaptive perplexity
    perplexity = min(perplexity, max(5, len(reps) // 100))

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
    unique_labels_sorted = np.sort(unique_labels)
    label_to_idx = {label: i for i, label in enumerate(unique_labels_sorted)}
    num_classes = len(unique_labels_sorted)

    # Map each sample to its color
    colors = [DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i in range(num_classes)]
    color_mapped = [colors[label_to_idx[l]] for l in all_labels]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        reps_2d[:, 0],
        reps_2d[:, 1],
        c=color_mapped,
        s=8,
        alpha=0.7,
    )
    
    ax.set_title(f"t-SNE Visualization - {unlearn_method}")
    #ax.axis('off')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    
    if num_classes <= 20:
        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=colors[i],
                markersize=8,
                label=str(unique_labels_sorted[i])
            )
            for i in range(num_classes)
        ]
        ax.legend(
            handles=legend_handles,
            title="Class",
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
        )
    
    plt.tight_layout()

    # Save figure
    save_path = save_path + f"visualize"
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path + f"/tsne_{unlearn_method}_{tag}.png", bbox_inches='tight')
    plt.close()

def linear_cka(X, Y, eps=1e-8):
    """
    X: (n, d1)
    Y: (n, d2)
    """

    # Center features
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    # Frobenius norm squared of cross-covariance
    numerator = torch.norm(X.T @ Y, p='fro') ** 2

    # Normalization
    denom = torch.norm(X.T @ X, p='fro') * torch.norm(Y.T @ Y, p='fro')

    cka = numerator / (denom + eps)

    return round(cka.item(), 4)

def representation_unlearning_score(cka_f, cka_r, original=False):
    # Compute harmonic mean between cka_f and cka_r
    if original:
        # If original, use 1 - cka_f for the harmonic mean calculation
        cka_f = 1 - cka_f

    rus = 2 * cka_f * cka_r / (cka_f + cka_r + 1e-8)
    return round(rus, 4)