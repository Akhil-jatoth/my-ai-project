import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import os


class MCNN(nn.Module):
    """
    Multi-Column CNN for Crowd Density Estimation
    
    Architecture:
    - 3 columns with different receptive fields
    - Each column processes the image at different scales
    - Outputs are concatenated and passed to density map generation
    """
    
    def __init__(self):
        super(MCNN, self).__init__()
        
        # Column 1: Large receptive field
        self.col1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
        )
        
        # Column 2: Medium receptive field
        self.col2 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(40, 20, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 10, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        
        # Column 3: Small receptive field
        self.col3 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Density map generation
        self.density_layer = nn.Sequential(
            nn.Conv2d(30, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        col1_out = self.col1(x)
        col2_out = self.col2(x)
        col3_out = self.col3(x)
        
        # Upsample outputs to original size
        col1_out = F.interpolate(col1_out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        col2_out = F.interpolate(col2_out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        col3_out = F.interpolate(col3_out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # Concatenate all columns
        combined = torch.cat([col1_out, col2_out, col3_out], dim=1)
        
        # Generate density map
        density_map = self.density_layer(combined)
        
        return density_map


class CrowdDensityPredictor:
    """
    Crowd Density Estimation Pipeline
    Handles image preprocessing, model inference, and density map generation
    """
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = MCNN().to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from {model_path}")
        else:
            print("Model initialized with random weights (training required)")
        
        self.model.eval()
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        img = Image.open(image_path).convert('RGB')
        
        # Resize to standard size
        img = img.resize((640, 480))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_tensor, img
    
    def predict(self, image_path):
        """
        Predict crowd density and count
        
        Returns:
            dict: Contains count, density_level, alert status, and density map
        """
        img_tensor, original_img = self.preprocess_image(image_path)
        
        with torch.no_grad():
            density_map = self.model(img_tensor)
        
        # Get crowd count
        crowd_count = int(density_map.sum().item())
        
        # Determine density level
        density_level = self._get_density_level(crowd_count)
        
        # Check overcrowding alert
        alert = crowd_count > 300  # Threshold for overcrowding
        
        # Get density map as numpy array
        density_array = density_map.squeeze().cpu().numpy()
        
        return {
            'count': crowd_count,
            'level': density_level,
            'alert': alert,
            'density_map': density_array,
            'original_image': original_img
        }
    
    def _get_density_level(self, count):
        """Classify density level based on crowd count"""
        if count < 50:
            return 'Low'
        elif count < 150:
            return 'Medium'
        elif count < 300:
            return 'High'
        else:
            return 'Very High'
    
    def generate_heatmap(self, density_map, output_path):
        """Generate and save heatmap visualization"""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Normalize density map
        density_normalized = (density_map - density_map.min()) / (density_map.max() - density_map.min() + 1e-5)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot density map
        im = ax2.imshow(density_normalized, cmap='jet')
        ax2.set_title('Density Map Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Plot original with overlay
        ax1.set_title('Crowd Count Visualization')
        ax1.imshow(np.ones_like(density_normalized) * 0.5, cmap='gray')
        ax1.imshow(density_normalized, cmap='jet', alpha=0.7)
        ax1.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to {output_path}")
