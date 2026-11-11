import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CLASSES AND FUNCTIONS (Must be at the top level)
# ---------------------------------------------------------
class KneeMRIDataset(Dataset):
    def __init__(self, dataframe, root_dir, label_cols, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = label_cols
        self.image_paths = []
        self.labels = []

        print("Scanning directory structure (this takes seconds)...")
        series_map = {}
        for root, _, files in os.walk(self.root_dir):
            valid_images = [os.path.join(root, f) for f in files if not f.startswith('.')]
            if valid_images:
                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part not in series_map:
                        series_map[part] = []
                    series_map[part].extend(valid_images)

        print("Mapping CSV labels to images...")
        found_count = 0
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Indexing"):
            series_uid = str(row.iloc[0])
            images = series_map.get(series_uid, [])

            if images:
                found_count += 1
                targets = row[self.label_cols].values.astype(np.float32)
                for img in images:
                    self.image_paths.append(img)
                    self.labels.append(targets)

        if found_count == 0:
            print("\nCRITICAL ERROR: No folder names matched the UIDs in the CSV.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label, os.path.basename(img_path)

def visualize_and_save(dataloader):
    print("Generating batch visualization...")
    images, labels, filenames = next(iter(dataloader))
    batch_size_plot = min(images.size(0), 4)

    fig, axes = plt.subplots(1, batch_size_plot, figsize=(15, 5))
    if batch_size_plot == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= batch_size_plot: break
        img = images[i].permute(1, 2, 0).cpu().numpy()
        # Corrected multiplication syntax here
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(f"Slice: {filenames[i][:8]}...\nLabel max: {labels[i].max().item():.1f}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("sample_mri_batch.png")
    plt.close()
    print("Saved 'sample_mri_batch.png'")


# ---------------------------------------------------------
# 2. EXECUTION BLOCK (Protects Mac multiprocessing)
# ---------------------------------------------------------
if __name__ == '__main__':
    DATA_DIR = "archive/train_series"
    CSV_PATH = "archive/pseudolabeled-train.csv"
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 1e-4

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    df = pd.read_csv(CSV_PATH)
    label_cols = df.columns[2:14]
    num_classes = len(label_cols)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = KneeMRIDataset(train_df, DATA_DIR, label_cols, transform)
    val_dataset = KneeMRIDataset(val_df, DATA_DIR, label_cols, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"\nTotal training slices: {len(train_dataset)} | Total validation slices: {len(val_dataset)}")

    if len(train_dataset) > 0:
        # Pass a temporary dataloader just for plotting
        plot_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        visualize_and_save(plot_loader)

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # TRAIN
        model.train()
        train_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # VALIDATE
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_resnet50_model.pth")
            print(f"--> Saved new best model with F1: {best_f1:.4f}")

    print("\nTraining complete. Best model saved as 'best_resnet50_model.pth'")