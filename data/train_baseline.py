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

# 1. Configuration
DATA_DIR = "data/archive/train_series"
CSV_PATH = "data/archive/train_series/pseudolabeled-train.csv"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data Preparation
df = pd.read_csv(CSV_PATH)
# Extract the binary abnormality columns based on the CSV structure
label_cols = df.columns[2:14]
num_classes = len(label_cols)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


class KneeMRIDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        print("Indexing dataset, this might take a moment...")
        for idx, row in self.dataframe.iterrows():
            series_uid = str(row.iloc[0])
            series_folder = glob.glob(f"{self.root_dir}/**/{series_uid}", recursive=True)

            if series_folder:
                images = glob.glob(f"{series_folder[0]}/*.png") + glob.glob(f"{series_folder[0]}/*.jpg")
                targets = row[label_cols].values.astype(np.float32)
                for img in images:
                    self.image_paths.append(img)
                    self.labels.append(targets)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


# Transform: Resize to 224x224 and normalize for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = KneeMRIDataset(train_df, DATA_DIR, transform)
val_dataset = KneeMRIDataset(val_df, DATA_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Total training slices: {len(train_dataset)}")
print(f"Total validation slices: {len(val_dataset)}")

# 3. Model Setup
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
# Replace final layer for our specific number of abnormality classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
best_f1 = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    # TRAIN
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
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
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # Apply sigmoid and threshold at 0.5 for F1 calculation
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Calculate Macro F1 Score
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")

    # Checkpoint saving
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_resnet50_model.pth")
        print(f"--> Saved new best model with F1: {best_f1:.4f}")

print("\nTraining complete. Best model saved as 'best_resnet50_model.pth'")