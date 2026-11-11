import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# WEEK 4: GRAD-CAM IMPLEMENTATION (Explainable AI)
# ---------------------------------------------------------
class ResNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # Hook the final convolutional layer of ResNet50
        target_layer = self.model.layer4[-1].conv3
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        outputs = self.model(x)
        
        if class_idx is None:
            # If no class specified, pick the one with highest prediction score
            class_idx = outputs.argmax(dim=1).item()
            
        self.model.zero_grad()
        # Backpropagate only the target class score
        score = outputs[0, class_idx]
        score.backward()

        # Weight the activations by the average gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU to only show positive influence
        cam = F.relu(cam)
        
        # Resize heatmap back to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        return cam, class_idx

# ---------------------------------------------------------
# EXECUTION SCRIPT
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Initializing Grad-CAM Interpretability Module...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    num_classes = 10 
    
    # 1. Load Model
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    try:
        model.load_state_dict(torch.load("best_resnet50_model.pth", map_location=device))
        print("Successfully loaded 'best_resnet50_model.pth'")
    except FileNotFoundError:
        print("Warning: Model weights not found. Using untrained weights for demonstration.")
        
    model = model.to(device)
    grad_cam = ResNetGradCAM(model)

    # 2. Get a dummy image or real image if possible
    # We will search the archive for any valid JPG to run inference on
    image_path = None
    for root, _, files in os.walk("archive/train_series"):
        for file in files:
            if not file.startswith('.'):
                image_path = os.path.join(root, file)
                break
        if image_path: break

    if image_path:
        print(f"Generating heatmap for slice: {os.path.basename(image_path)}")
        original_img = Image.open(image_path).convert('RGB').resize((224, 224))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(original_img).unsqueeze(0).to(device)
        
        # 3. Generate Heatmap
        heatmap, pred_class = grad_cam(input_tensor)
        
        # 4. Plot and Save Overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original MRI Slice")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Network Attention (Grad-CAM)")
        axes[1].axis('off')
        
        axes[2].imshow(original_img)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.5) # Overlay
        axes[2].set_title(f"Clinical Overlay\n(Model focus for class {pred_class})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig("gradcam_proof.png")
        print("Saved Explainable AI proof to 'gradcam_proof.png'")
    else:
        print("Could not find an image in 'archive/train_series' to run Grad-CAM on.")