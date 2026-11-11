import os
import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# Define paths based on the project structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'archive')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'pseudolabeled-train.csv')
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train_series')

class RSNAKneeDataset(Dataset):
    """
    Custom PyTorch Dataset mapped to handle nested folder structures
    (train_series/study/series/image.jpg) using glob.
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # StudyInstanceUID
        self.study_ids = self.data_frame.iloc[:, 0].astype(str).values
        # 10 Pseudo-labels columns
        self.labels = self.data_frame.iloc[:, 2:12].values 

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        study_folder = os.path.join(self.img_dir, study_id)
        
        try:
            labels = torch.tensor(self.labels[idx].astype(float), dtype=torch.float32)
        except ValueError:
            labels = torch.zeros(10, dtype=torch.float32)

        # Use glob to dynamically search through nested series folders for images
        search_pattern = os.path.join(study_folder, '**', '*.[jJ][pP][gG]')
        image_paths = glob.glob(search_pattern, recursive=True)
        
        # For the baseline model, we'll grab the middle slice of the series if available
        # (Handling full 3D volumes is a next-step optimization)
        selected_image = torch.zeros((3, 224, 224)) # default blank
        
        if len(image_paths) > 0:
            middle_idx = len(image_paths) // 2
            try:
                image = Image.open(image_paths[middle_idx]).convert('RGB')
                if self.transform:
                    selected_image = self.transform(image)
            except Exception as e:
                print(f"Error loading {image_paths[middle_idx]}: {e}")

        return {'study_id': study_id, 'image': selected_image, 'labels': labels}

class RSNAKneeResNet(nn.Module):
    """
    Baseline Multi-label Classification Model using a pretrained ResNet50 backbone.
    """
    def __init__(self, num_classes=10):
        super(RSNAKneeResNet, self).__init__()
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Freeze the early layers to speed up initial training (transfer learning)
        for param in list(self.backbone.parameters())[:-15]: 
            param.requires_grad = False
            
        # Replace the final fully connected layer to output our 10 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
            # Note: BCEWithLogitsLoss will handle the sigmoid activation during training
        )

    def forward(self, x):
        return self.backbone(x)

def visualize_batch(dataloader):
    """
    Pulls a batch from the dataloader and plots the images using Matplotlib.
    """
    print("Generating batch visualization...")
    batch = next(iter(dataloader))
    images, labels, ids = batch['image'], batch['labels'], batch['study_id']
    
    batch_size = images.size(0)
    fig, axes = plt.subplots(1, min(batch_size, 4), figsize=(15, 5))
    
    if batch_size == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        if i >= batch_size: break
        # Un-normalize image for visualization (assuming ImageNet norms)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)
        
        ax.imshow(img)
        ax.set_title(f"Study: {ids[i][:8]}...\nLabels max: {labels[i].max().item():.1f}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("--- RSNA Knee Abnormality Pipeline Initialized ---")
    
    # Standard ResNet image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Initialize Dataset & DataLoader
    if os.path.exists(TRAIN_CSV_PATH):
        dataset = RSNAKneeDataset(TRAIN_CSV_PATH, TRAIN_IMAGES_DIR, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"Dataset mapped successfully. Total studies: {len(dataset)}")
        
        # 2. Visualize a batch (EDA)
        if len(dataset) > 0:
            visualize_batch(dataloader)
            
        # 3. Initialize Model
        print("\nInitializing ResNet50 Backbone...")
        model = RSNAKneeResNet(num_classes=10)
        print(f"Model architecture ready. Output shape for a batch of 4: {model(torch.randn(4, 3, 224, 224)).shape}")
        print("Ready for training loop implementation next week!")
    else:
        print(f"Warning: {TRAIN_CSV_PATH} not found. Ensure data is extracted.")