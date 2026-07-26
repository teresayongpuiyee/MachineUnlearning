import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset
from sklearn.neighbors import KNeighborsClassifier
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from typing import Tuple, Optional, Sequence
from torch.nn import functional as F
from unlearn_strategies import utils

# t-SNE visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from openTSNE import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import warnings
import random

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
            # TODO: flag all_layer and index layer of interest as feat
            feat = model.feature_extractor(data)
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
    seed: int = 42
) -> float:
    # Subsampling of retain data
    target_size = test_reps.shape[0]

    indices = np.arange(len(retain_reps))
    _, sampled_indices = train_test_split(
        indices,
        test_size=target_size,
        stratify=retain_labels.numpy(),
        random_state=seed
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
        random_state=seed
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
    seed: int = 42
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
            random_state=seed
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
        random_state=seed
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
    unlearn_class: int,
    seed: int = 42
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
            random_state=seed
        )
        train_reps = train_reps[sampled_indices]
        train_labels = train_labels[sampled_indices]
    
    # Prepare data for attack
    X_full = torch.cat([train_reps, test_reps], dim=0).numpy()
    X_labels = np.concatenate([train_labels.numpy(), test_labels.numpy()])
    y_full = np.concatenate([np.ones(len(train_reps)), np.zeros(len(test_reps))])
    
    strat_key = np.array([f"{lbl}_{mem}" for lbl, mem in zip(X_labels, y_full)])

    # Five-fold cross-validation attack
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

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
        "train_acc": float(np.mean(results["train_acc"])* 100),
        "train_f1": float(np.mean(results["train_f1"])* 100),
        "test_acc": float(np.mean(results["test_acc"])* 100),
        "test_f1": float(np.mean(results["test_f1"])* 100),
        "forget_fpr": float(np.mean(results["forget_fpr"])* 100),
    }

    forget_asr =None
    if results["forget_asr"]:
        forget_asr = float(np.mean(results["forget_asr"]) * 100)

    return metrics_dict, forget_asr


# MIA in Representation Space based on SURE: https://openreview.net/forum?id=KzSGJy1PIf
def sure_miars(
    train_reps: torch.tensor,
    test_reps: torch.tensor,
    train_labels: torch.tensor,
    test_labels: torch.tensor,
    unlearn_class: int,
    n_neighbors: int = 5,
    seed: int = 42
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
            random_state=seed
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
        random_state=seed
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

def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
 
 
def _collect_features_np(
    loader: Optional[DataLoader],
    model: torch.nn.Module,
    n_passes: int = 1,
    seed: Optional[int] = None,
) -> Optional[tuple]:
    """Stack features + labels over n_passes of `loader` via get_representations().
    Augmentation (if in the loader's transform) is resampled per pass -- passes
    advance the RNG naturally, so reproducibility comes from your runner's global
    seed. If `seed` is given, reseed per pass (seed + p). None => leave RNG alone.
    Returns numpy arrays, or None if loader is None."""
    if loader is None:
        return None
    Xs, ys = [], []
    for p in range(n_passes):
        if seed is not None:
            _seed_all(seed + p)
        X, y = get_representations(loader, model)   # torch tensors on CPU
        Xs.append(X.numpy())
        ys.append(y.numpy())
    return np.concatenate(Xs, 0), np.concatenate(ys, 0)
 
 
def _make_lr(C: float, tol: float, max_iter: int, fit_intercept: bool) -> LogisticRegression:
    kwargs = dict(penalty="l2", C=C, solver="lbfgs", tol=tol,
                  max_iter=max_iter, fit_intercept=fit_intercept)
    # multinomial is the L-BFGS default in modern sklearn; force it where the kwarg
    # still exists (<1.7) and fall back where it was removed (>=1.7 raises TypeError).
    try:
        return LogisticRegression(multi_class="multinomial", **kwargs)
    except TypeError:
        return LogisticRegression(**kwargs)
 
 
def logistic_probe_lbfgs(
    train_loader: DataLoader,
    retain_eval_loader: DataLoader,        # train split, clean -> train-set decodability
    unlearn_eval_loader: DataLoader,       # train split, clean
    model: torch.nn.Module,
    test_retain_loader: Optional[DataLoader] = None,   # test split -> generalization
    test_unlearn_loader: Optional[DataLoader] = None,
    C: float = 1.0,                 # L2 strength; lambda = 1/C
    tol: float = 1e-4,
    max_iter: int = 5000,
    fit_intercept: bool = True,
    n_aug_passes: int = 1,
    seed: Optional[int] = None,     # leave None: your runner owns global seeding
    val_loader: Optional[DataLoader] = None,   # C selection only; disjoint from all report sets
    C_grid: Optional[Sequence[float]] = None,
) -> dict:
    """One L-BFGS probe fit on augmented train features; reports the four splits
    your SGD probe reports. Use ONE fixed C across all models."""
    X_tr, y_tr = _collect_features_np(train_loader, model, n_passes=n_aug_passes, seed=seed)
 
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
 
    if val_loader is not None and C_grid:
        Xv, yv = _collect_features_np(val_loader, model, n_passes=1, seed=None)
        Xv_s = scaler.transform(Xv)
        best_C, best_acc = C, -1.0
        for c in C_grid:
            acc_c = _make_lr(c, tol, max_iter, fit_intercept).fit(X_tr_s, y_tr).score(Xv_s, yv)
            if acc_c > best_acc:
                best_acc, best_C = acc_c, c
        C = best_C
 
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = _make_lr(C, tol, max_iter, fit_intercept).fit(X_tr_s, y_tr)
    n_iter = int(np.max(clf.n_iter_))
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
 
    def _acc(loader):
        data = _collect_features_np(loader, model, n_passes=1, seed=None)  # clean, single pass
        if data is None:
            return None
        X, y = data
        return 100.0 * clf.score(scaler.transform(X), y)
 
    return {
        # train-set decodability (Table 12 Acc_probe_u / Acc_probe_r); primary band quantity
        "forget_accuracy": _acc(unlearn_eval_loader),
        "retain_accuracy": _acc(retain_eval_loader),
        # test-set generalization
        "test_forget_accuracy": _acc(test_unlearn_loader),
        "test_retain_accuracy": _acc(test_retain_loader),
        # ---- four items to report explicitly ----
        "fit_intercept": fit_intercept, "C": C, "l2_lambda": 1.0 / C,
        "tol": tol, "max_iter": max_iter,
        # ---- protocol + convergence ----
        "n_aug_passes": n_aug_passes, "seed": seed,
        "n_iter": n_iter, "converged": converged,
        "solver": "lbfgs", "multi_class": "multinomial",
    }


def linear_probing(
    train_loader: DataLoader,
    test_loader: DataLoader,
    retain_eval_loader: DataLoader,
    unlearn_eval_loader: DataLoader,
    test_retain_loader: DataLoader,
    test_unlearn_loader: DataLoader,
    model: torch.nn.Module,
    num_classes: int,
    epochs: int = 10,
    lr: float = 1e-3,
    momentum: float = 0.0,
    patience: int = 0,
    batch_size: int = 128,
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
    head = nn.Linear(feat.size(1), num_classes).to(device)
    nn.init.xavier_normal_(head.weight)
    nn.init.zeros_(head.bias)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    for param in head.parameters():
        param.requires_grad = True

    train_loader = DataLoader(
        train_loader.dataset, batch_size=batch_size, shuffle=True, num_workers=train_loader.num_workers, pin_memory=True, persistent_workers=True
    )
    
    optimizer = optim.SGD(head.parameters(), lr=lr, momentum= momentum)
    #optimizer = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    def eval_metrics(loader):
        head.eval()
        correct = 0
        total = 0
        loss_sum = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                feat = model.feature_extractor(x)
                logits = head(feat)
                loss = F.cross_entropy(logits, y, reduction='sum')
                loss_sum += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100.0 * correct / total if total > 0 else 0.0
        loss = loss_sum / total if total > 0 else 0.0
        return acc, loss

    best_test_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    log_dict = []

    # Train linear head
    for epoch in range(epochs):
        loss_list = []
        head.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                feat = model.feature_extractor(x)
            logits = head(feat)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        #scheduler.step()
        train_loss = np.mean(np.array(loss_list))
        retain_acc, _ = eval_metrics(retain_eval_loader)
        forget_acc, _ = eval_metrics(unlearn_eval_loader)
        _, test_loss = eval_metrics(test_loader)
        test_retain_acc, _ = eval_metrics(test_retain_loader)
        test_forget_acc, _ = eval_metrics(test_unlearn_loader)

        log_dict.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "forget_acc": forget_acc,
            "retain_acc": retain_acc,
            "test_retain_acc": test_retain_acc,
            "test_forget_acc": test_forget_acc,
        })

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                break

    return {
        "best_epoch": best_epoch,
    }, log_dict

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
    seed: int = 42
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
            random_state=seed
        )

    # Standardize
    reps = StandardScaler().fit_transform(reps)

    # PCA for speed (retain 50 components or less if input dim < 50)
    if reps.shape[1] > 50:
        reps = PCA(n_components=50, random_state=seed).fit_transform(reps)

    # Adaptive perplexity
    perplexity = min(perplexity, max(5, len(reps) // 100))

    # Fast t-SNE using openTSNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        metric="euclidean",
        random_state=seed,
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

    return cka.item()


def svcca(X, Y, var_threshold=0.99, epsilon=1e-10):
    """
    Compute the SVCCA similarity between two representations of shape (n, d).

    Two stages:
      1. SVD: reduce each representation to the top singular directions
         retaining `var_threshold` of variance (strips low-variance noise dims).
      2. CCA: find maximally-correlated linear projections of the two reduced
         reps. The mean of the canonical correlations is the SVCCA score.

    X : (n, d1), Y : (n, d2)  — same n examples, features as columns; d1 may != d2.
    Returns (mean_correlation, canonical_correlations_sorted_desc).
    """
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must share axis 0 (number of examples).")

    # Center each feature across examples (required for CCA).
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)

    Xr = _svd_reduce(X, var_threshold)
    Yr = _svd_reduce(Y, var_threshold)
    corrs = _cca_correlations(Xr, Yr, epsilon)
    return float(corrs.mean()), corrs


def _svd_reduce(A, var_threshold):
    """Project centered A (n,d) onto top PCs retaining var_threshold of variance."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    if var_threshold is None or var_threshold >= 1.0:
        k = len(s)
    else:
        var = s ** 2; total = var.sum()
        if total <= 0:
            return np.zeros((A.shape[0], 1))
        k = int(np.searchsorted(np.cumsum(var) / total, var_threshold) + 1)
        k = max(1, min(k, len(s)))
    return U[:, :k] * s[:k]   # scores in retained PC basis == A @ V_k


def _cca_correlations(X, Y, epsilon=1e-10):
    """Canonical correlations via whitening: singular values of the
    whitened cross-covariance are exactly the canonical correlations."""
    n = X.shape[0]
    Sxx, Syy, Sxy = (X.T @ X)/n, (Y.T @ Y)/n, (X.T @ Y)/n
    T = _inv_sqrt_psd(Sxx, epsilon) @ Sxy @ _inv_sqrt_psd(Syy, epsilon)
    return np.clip(np.linalg.svd(T, compute_uv=False), 0.0, 1.0)


def _inv_sqrt_psd(M, epsilon):
    """Inverse symmetric sqrt of a PSD matrix; drops eigenvalues < epsilon
    so it stays stable when M is rank-deficient (common after SVD truncation)."""
    M = (M + M.T) / 2.0
    vals, vecs = np.linalg.eigh(M)
    keep = vals > epsilon
    vals, vecs = vals[keep], vecs[:, keep]
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def representation_unlearning_score(cka_f, cka_r, original=False):
    # Compute harmonic mean between cka_f and cka_r
    if original:
        # If original, use 1 - cka_f for the harmonic mean calculation
        cka_f = 1 - cka_f

    rus = 2 * cka_f * cka_r / (cka_f + cka_r + 1e-8)
    return rus


from itertools import chain

def _endless(loader):
    # re-iterates the retain loader forever (reshuffles each pass, no caching)
    while True:
        for batch in loader:
            yield batch


class BalancedRelearnLoader:
    """Yields mixed batches: f forget + r*f retain. Epoch length = forget passes."""
    def __init__(self, forget_loader, retain_loader, r):
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.r = r
        self.dataset = forget_loader.dataset      # epoch tracks the forget set
        self.batch_size = forget_loader.batch_size

    def __len__(self):
        return len(self.forget_loader)            # = ceil(k_f / f) steps per epoch

    def __iter__(self):
        retain_it = _endless(self.retain_loader)
        for fx, fy in self.forget_loader:         # fx: f (or ragged last)
            rx, ry = next(retain_it)              # rx: r*f
            k = self.r * fx.size(0)               # keep exact ratio on ragged batch
            x = torch.cat([fx, rx[:k]], dim=0)    # total = f*(1+r)
            y = torch.cat([fy, ry[:k]], dim=0)
            yield x, y


def relearning_attack(
        logger,
        model,
        unlearn_loader: DataLoader,
        retain_loader: DataLoader,
        test_loader: DataLoader,
        sample_size: int,
        epoch: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        device: str,
        seed: int = 42,
        model_name: str = "model",
        save_dir: str = ".",
        retain_per_forget=None,          # r: None/0 -> forget-only; e.g. 9 -> balanced
):
    # Seeded generator so sampling (and the loader shuffle) is reproducible.
    generator = torch.Generator().manual_seed(seed)
    g_forget = torch.Generator().manual_seed(seed + 1)
    g_retain = torch.Generator().manual_seed(seed + 2)

    # sample from unlearn_loader dataset only based on sample_size
    unlearn_ds = unlearn_loader.dataset
    n = min(sample_size, len(unlearn_ds))
    sampled_idx = torch.randperm(len(unlearn_ds), generator=generator)[:n]
    relearning_ds = Subset(unlearn_ds, sampled_idx.tolist())

    f = min(sample_size, unlearn_loader.batch_size)

    # construct relearning loader from sampled dataset
    forget_loader = DataLoader(
        relearning_ds,
        batch_size=f,
        shuffle=True,
        num_workers=unlearn_loader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        generator=g_forget,
        drop_last=False,
    )

    # ---- pick the train loader for the chosen variant ----
    if retain_per_forget:                              # balanced retain-rehearsal
        retain_ds = retain_loader.dataset             # D_r train (disjoint from retain-test)
        retain_ft_loader = DataLoader(
            retain_ds, batch_size=retain_per_forget * f, shuffle=True, drop_last=True,
            num_workers=retain_loader.num_workers, pin_memory=True,
            persistent_workers=True, generator=g_retain,
        )
        relearning_loader = BalancedRelearnLoader(forget_loader, retain_ft_loader, retain_per_forget)
    else:                                              # forget-only (your current variant)
        relearning_loader = forget_loader

    # Fine tune model
    ft_model, log_dict = utils.training_optimization(
        logger,
        model= model,
        train_loader= relearning_loader,
        test_loader= test_loader,
        epochs= epoch,
        device= device,
        desc= "Relearning attack",
        opt="sgd",
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        unlearn_loader= unlearn_loader,
        retain_loader= retain_loader,
    )

    tag = f"_r{retain_per_forget}" if retain_per_forget else ""

    # write dict to csv file, naming model_name+seed+sample_size
    csv_path = f"{save_dir}{model_name}_seed{seed}_size{sample_size}{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_dict[0].keys()))
        writer.writeheader()
        writer.writerows(log_dict)
    logger.info(f"Saved relearning attack log to {csv_path}")

    return ft_model, log_dict