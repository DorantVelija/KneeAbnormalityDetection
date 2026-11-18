import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def to_long_path(path):
    abs_path = os.path.abspath(path)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def resolve_data_paths():
    candidates = [
        ("archive/train_series", "archive/pseudolabeled-train.csv"),
        ("data/archive/train_series", "data/archive/pseudolabeled-train.csv"),
    ]
    for img_dir, csv_path in candidates:
        if os.path.exists(csv_path) and os.path.exists(img_dir):
            return img_dir, csv_path
    raise FileNotFoundError("Could not locate train_series and pseudolabeled-train.csv.")


def build_series_index(root_dir):
    series_map = {}
    for root, _, files in os.walk(to_long_path(root_dir)):
        valid_images = sorted([
            os.path.join(root, f) for f in files if not f.startswith(".")
        ])
        if valid_images:
            clean_root = root.replace("\\\\?\\", "")
            for part in clean_root.split(os.sep):
                if part not in series_map:
                    series_map[part] = []
                series_map[part].extend(valid_images)
    return series_map


class DiagnosticKneeDataset(Dataset):
    def __init__(self, dataframe, series_map, label_cols, transform=None, central_slices=5):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.groups = []

        for _, row in dataframe.iterrows():
            uid = str(row.iloc[0])
            images = series_map.get(uid, [])
            if not images:
                continue

            # Keep central diagnostic window to eliminate empty peripheral MRI slices
            if central_slices and len(images) > central_slices:
                mid = len(images) // 2
                half = central_slices // 2
                images = images[max(0, mid - half): mid + half + 1]

            targets = row[label_cols].values.astype(np.float32)
            for img_path in images:
                self.image_paths.append(img_path)
                self.labels.append(targets)
                self.groups.append(uid)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label


def create_model(num_classes, device):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def evaluate_predictions(y_true, y_prob, label_cols):
    rows = []
    optimal_thresholds = []

    for i, col in enumerate(label_cols):
        yt = y_true[:, i]
        yp = y_prob[:, i]

        try:
            auroc = roc_auc_score(yt, yp)
        except ValueError:
            auroc = np.nan

        try:
            prauc = average_precision_score(yt, yp)
        except ValueError:
            prauc = np.nan

        # Find threshold maximizing F1 for this specific pathology
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.15, 0.85, 0.05):
            score = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t

        default_f1 = f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
        brier = brier_score_loss(yt, yp)
        optimal_thresholds.append(best_t)

        rows.append({
            "Pathology": col,
            "Support_Positives": int(yt.sum()),
            "AUROC": round(float(auroc), 4) if not np.isnan(auroc) else 0.0,
            "PR_AUC": round(float(prauc), 4) if not np.isnan(prauc) else 0.0,
            "Default_F1_0.5": round(float(default_f1), 4),
            "Optimal_Threshold": round(float(best_t), 2),
            "Calibrated_F1": round(float(best_f1), 4),
            "Brier_Score": round(float(brier), 4),
        })

    metrics_df = pd.DataFrame(rows)
    return metrics_df, np.array(optimal_thresholds)


def plot_roc_and_pr(y_true, y_prob, label_cols, save_path="validity_roc_pr_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(label_cols)))

    for i, (col, color) in enumerate(zip(label_cols, colors)):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            continue

        fpr, tpr, _ = roc_curve(yt, yp)
        auroc = roc_auc_score(yt, yp)
        ax1.plot(fpr, tpr, color=color, lw=1.8, label=f"{col[:18]} (AUC={auroc:.3f})")

        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax2.plot(rec, prec, color=color, lw=1.8, label=f"{col[:18]} (AP={ap:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_title("Multi-Label ROC Curves (Discrimination Validity)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_title("Precision-Recall Curves (Imbalance Validity)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved '{save_path}'")


def plot_calibration(y_true, y_prob, save_path="validity_calibration_curve.png"):
    prob_true, prob_pred = calibration_curve(y_true.ravel(), y_prob.ravel(), n_bins=10, strategy="uniform")
    overall_brier = brier_score_loss(y_true.ravel(), y_prob.ravel())

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Clinical Calibration")
    plt.plot(prob_pred, prob_true, "s-", color="#d62728", lw=2, label=f"ResNet50 (Brier Score = {overall_brier:.4f})")
    plt.title("Probability Calibration Curve (Reliability Validity)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Fraction")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved '{save_path}'")


def plot_kfold_results(fold_metrics, save_path="validity_kfold_boxplot.png"):
    df_folds = pd.DataFrame(fold_metrics)
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(df_folds))
    width = 0.35

    rects1 = ax.bar(x - width / 2, df_folds["Macro_AUROC"], width, label="Macro AUROC", color="#1f77b4")
    rects2 = ax.bar(x + width / 2, df_folds["Calibrated_Macro_F1"], width, label="Calibrated Macro F1", color="#2ca02c")

    mean_auc = df_folds["Macro_AUROC"].mean()
    std_auc = df_folds["Macro_AUROC"].std()
    mean_f1 = df_folds["Calibrated_Macro_F1"].mean()
    std_f1 = df_folds["Calibrated_Macro_F1"].std()

    ax.axhline(mean_auc, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Mean AUROC: {mean_auc:.3f} ± {std_auc:.3f}")
    ax.axhline(mean_f1, color="#2ca02c", linestyle="--", alpha=0.7, label=f"Mean F1: {mean_f1:.3f} ± {std_f1:.3f}")

    ax.set_ylabel("Score")
    ax.set_title("5-Fold Study-Grouped Cross-Validation Stability")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in df_folds["Fold"]])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved '{save_path}'")


def run_inference(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())
    return np.vstack(all_targets), np.vstack(all_probs)


if __name__ == "__main__":
    DATA_DIR, CSV_PATH = resolve_data_paths()
    BATCH_SIZE = 64
    KFOLD_EPOCHS = 2
    LEARNING_RATE = 1e-4
    CENTRAL_SLICES = 5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Running Validity Suite on device: {device}")

    df = pd.read_csv(CSV_PATH)
    label_cols = df.columns[2:14].tolist()
    num_classes = len(label_cols)
    groups = df.iloc[:, 0].astype(str).values

    print("Building single-pass directory index...")
    series_map = build_series_index(DATA_DIR)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = 4 if os.name == "nt" else 2
    pin_mem = device.type == "cuda"

    # --- PART 1: Audit Existing Model Checkpoint (if available) ---
    if os.path.exists("best_resnet50_model.pth"):
        print("\n[1/2] Auditing existing 'best_resnet50_model.pth' on leak-free holdout split...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        _, val_idx = next(gss.split(df, groups=groups))
        holdout_dataset = DiagnosticKneeDataset(df.iloc[val_idx], series_map, label_cols, eval_transform, CENTRAL_SLICES)
        holdout_loader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

        audit_model = create_model(num_classes, device)
        audit_model.load_state_dict(torch.load("best_resnet50_model.pth", map_location=device, weights_only=True))

        y_true, y_prob = run_inference(audit_model, holdout_loader, device)
        metrics_df, opt_thresh = evaluate_predictions(y_true, y_prob, label_cols)
        metrics_df.to_csv("validity_metrics_table.csv", index=False)
        print("\nPer-Class Statistical Validity Audit:")
        print(metrics_df.to_string(index=False))

        plot_roc_and_pr(y_true, y_prob, label_cols)
        plot_calibration(y_true, y_prob)

    # --- PART 2: 5-Fold Study-Grouped Cross-Validation ---
    print("\n[2/2] Running 5-Fold Study-Grouped Cross-Validation (Zero Patient/Study Leakage)...")
    gkf = GroupKFold(n_splits=5)
    fold_results = []
    best_overall_auc = 0.0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), start=1):
        print(f"\n--- Fold {fold}/5 ---")
        train_ds = DiagnosticKneeDataset(df.iloc[train_idx], series_map, label_cols, train_transform, CENTRAL_SLICES)
        val_ds = DiagnosticKneeDataset(df.iloc[val_idx], series_map, label_cols, eval_transform, CENTRAL_SLICES)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

        model = create_model(num_classes, device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        for epoch in range(KFOLD_EPOCHS):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{KFOLD_EPOCHS}", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)

        y_true, y_prob = run_inference(model, val_loader, device)
        fold_df, opt_thresh = evaluate_predictions(y_true, y_prob, label_cols)

        macro_auc = fold_df["AUROC"].mean()
        macro_pr = fold_df["PR_AUC"].mean()
        cal_f1 = fold_df["Calibrated_F1"].mean()
        brier = fold_df["Brier_Score"].mean()

        print(f"Fold {fold} Summary -> Macro AUROC: {macro_auc:.4f} | Macro PR-AUC: {macro_pr:.4f} | Calibrated F1: {cal_f1:.4f} | Brier: {brier:.4f}")

        fold_results.append({
            "Fold": fold,
            "Macro_AUROC": macro_auc,
            "Macro_PR_AUC": macro_pr,
            "Calibrated_Macro_F1": cal_f1,
            "Mean_Brier_Score": brier,
        })

        if macro_auc > best_overall_auc:
            best_overall_auc = macro_auc
            torch.save(model.state_dict(), "best_kfold_resnet50.pth")

    summary_df = pd.DataFrame(fold_results)
    summary_df.to_csv("validity_kfold_summary.csv", index=False)
    plot_kfold_results(fold_results)

    print("\n=== FINAL 5-FOLD VALIDITY REPORT ===")
    print(f"Macro AUROC:         {summary_df['Macro_AUROC'].mean():.4f} ± {summary_df['Macro_AUROC'].std():.4f}")
    print(f"Macro PR-AUC:        {summary_df['Macro_PR_AUC'].mean():.4f} ± {summary_df['Macro_PR_AUC'].std():.4f}")
    print(f"Calibrated Macro F1: {summary_df['Calibrated_Macro_F1'].mean():.4f} ± {summary_df['Calibrated_Macro_F1'].std():.4f}")
    print(f"Mean Brier Score:    {summary_df['Mean_Brier_Score'].mean():.4f}")