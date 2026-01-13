import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from model import models
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score

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

# Representation-level Membership Inference Attack (MIA)
def repr_mia(
    retain_reps: torch.tensor,
    forget_reps: torch.tensor,
    test_reps: torch.tensor,
) -> float:
    # Prepare data for attack: retain (member, label=1), test (non-member, label=0)
    X_full = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y_full = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    X, X_test, y, y_test = train_test_split(
        X_full, 
        y_full, 
        test_size=0.2,           # 20% for evaluating the attack
        stratify=y_full,          # Essential: maintains the 50/50 member/non-member ratio
        random_state=42
    )

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
    return metrics_dict, round(asr, 4)

# Representation-level Membership Inference Attack (MIA) using five-fold attack and linear regressor
# based on https://arxiv.org/abs/2511.19339 
def representation_level_mia(
    retain_reps: torch.tensor,
    forget_reps: torch.tensor,
    test_reps: torch.tensor,
) -> float:
    """
    Perform a five-fold membership inference attack on the forget set using representations from a specified layer.
    Args:
        retain_reps: Representations for the retain (train) set
        forget_reps: Representations for the forget set (attack target)
        test_reps: Representations for the test set (non-member)
    Returns:
        Attack model metrics (dict)
        Attack success rate (float, percent)
    """    
    # Prepare data for attack: retain (member, label=1), test (non-member, label=0)
    X = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    # Five-fold cross-validation attack
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_acc_list = []
    train_f1_list = []
    test_acc_list = []
    test_f1_list = []
    asr_list = []
    for train_idx, test_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        forget_X = scaler.transform(forget_reps.numpy())

        clf = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train) * 100
        train_preds = clf.predict(X_train)
        train_f1 = f1_score(y_train, train_preds, average="macro") * 100

        test_acc = clf.score(X_test, y_test) * 100
        test_preds = clf.predict(X_test)
        test_f1 = f1_score(y_test, test_preds, average="macro") * 100
        
        # Attack on forget set (should be members)
        forget_pred = clf.predict(forget_X)
        asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
        
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)
        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)
        asr_list.append(asr)

    metrics_dict = {
        "train_acc": round(float(np.mean(train_acc_list)), 4),
        "train_f1": round(float(np.mean(train_f1_list)), 4),
        "test_acc": round(float(np.mean(test_acc_list)), 4),
        "test_f1": round(float(np.mean(test_f1_list)), 4),
    }

    return metrics_dict, round(np.mean(asr_list), 4)


# MIA in Representation Space based on https://openreview.net/forum?id=KzSGJy1PIf
def miars(
    retain_reps: torch.tensor,
    test_reps: torch.tensor,
    forget_reps: torch.tensor,
    n_neighbors: int = 5
) -> float:
    """
    Trains a KNN classifier to distinguish between train and test samples based on their representations,
    then applies the trained KNN to classify the forget samples and calculates the attack success rate (ASR).
    Args:
        retain_reps: Representations for the train set
        test_reps: Representations for the test set
        forget_reps: Representations for the forget set
        n_neighbors: Number of neighbors for KNN
    Returns:
        Attack model metrics (dict)
        Attack success rate (float, percent of forget samples classified as train/member)
    """
    X_full = torch.cat([retain_reps, test_reps], dim=0).numpy()
    y_full = np.concatenate([np.ones(len(retain_reps)), np.zeros(len(test_reps))])

    X, X_test, y, y_test = train_test_split(
        X_full, 
        y_full, 
        test_size=0.2,           # 20% for evaluating the attack
        stratify=y_full,          # Essential: maintains the 50/50 member/non-member ratio
        random_state=42
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    forget_X = scaler.transform(forget_reps.numpy())

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)

    train_acc = knn.score(X, y) * 100
    train_preds = knn.predict(X)
    train_f1 = f1_score(y, train_preds, average="macro") * 100

    test_acc = knn.score(X_test, y_test) * 100
    test_preds = knn.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro") * 100

    metrics_dict = {
        "train_acc": round(float(train_acc), 4),
        "train_f1": round(float(train_f1), 4),
        "test_acc": round(float(test_acc), 4),
        "test_f1": round(float(test_f1), 4),
    }

    forget_pred = knn.predict(forget_X)
    asr = forget_pred.mean() * 100  # percent of forget samples predicted as member
    return metrics_dict, round(asr, 4)

# MIA using MLP
def mia_mlp(
    retain_reps: torch.tensor,
    test_reps: torch.tensor,
    forget_reps: torch.tensor,
    device: torch.device,
    num_epochs: int = 50,
    logger = None,
) -> float:
    X_train_tensor = torch.cat([retain_reps, test_reps], dim=0)
    Y_train_tensor = torch.cat([torch.ones(len(retain_reps)), torch.zeros(len(test_reps))]).unsqueeze(1)

    X_val_tensor = forget_reps
    Y_val_tensor = torch.zeros(len(X_val_tensor)).unsqueeze(1)

    # Create Datasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = models.AttackMLP(
        input_size=512, 
        hidden_1=128,
        hidden_2=64,
        output_size=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    def evaluate(val_loader: DataLoader, model: torch.nn.Module, device: torch.device) -> float:
        model.eval() # Set model to evaluation mode
        correct_predictions = 0
        total_samples = 0
        
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                logits = model(features)
                
                # Apply Sigmoid and threshold to get binary prediction (0 or 1)
                probabilities = torch.sigmoid(logits)
                predictions = (probabilities >= 0.5).float()
                
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_predictions, average="macro") * 100
        accuracy = (correct_predictions / total_samples) * 100.0
        return round(accuracy, 4), round(f1, 4)
    
    train_acc, train_f1 = evaluate(train_loader, model, device)
    forget_acc, _ = evaluate(val_loader, model, device)

    metrics_dict = {
        "train_acc": float(train_acc),
        "train_f1": float(train_f1),
    }

    return metrics_dict, forget_acc

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
    reps_2d = tsne.fit_transform(reps)

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