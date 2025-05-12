import os
import io
import base64
import json
import logging
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
import cv2
from waitress import serve

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max image size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# -------------- Model Definitions --------------

class SteganographyCNN(nn.Module):
    """Deep learning model for steganography detection."""
    
    def __init__(self):
        super(SteganographyCNN, self).__init__()
                # Rich feature extraction
        self.features = nn.Sequential(
            # First block - detect basic artifacts
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block - detect more complex patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block - high-level features
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)  # Three classes: no steganography, LSB steganography, DCT steganography
        )
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
        def detect(self, image):
            """
            Full detection pipeline
            Returns dict with probabilities and detected method
            """
            self.eval()
            with torch.no_grad():
                output = self.forward(image)
                probabilities = torch.softmax(output, dim=1)[0]
                    
                # Get prediction
                pred_class = torch.argmax(probabilities).item()
                confidence = probabilities[pred_class].item() * 100
                
                # Map class index to method name
                methods = ["none", "lsb", "dct"]
                detected_method = methods[pred_class]
                
                return {
                    "hasHiddenData": detected_method != "none",
                    "confidence": round(confidence, 2),
                    "method": detected_method if detected_method != "none" else None,
                    "probabilities": {
                        "none": round(float(probabilities[0]) * 100, 2),
                        "lsb": round(float(probabilities[1]) * 100, 2),
                        "dct": round(float(probabilities[2]) * 100, 2)
                    }
                }
class BusyAreaDetector:
    """Detects visually busy or complex regions in an image."""
    def __init__(self):
        # Sensitivity presets (adjust based on empirical testing)
        self.sensitivity_presets = {
            'low': {'edge_threshold': 120, 'density_threshold': 0.3, 'sigma': 2.0},
            'medium': {'edge_threshold': 100, 'density_threshold': 0.2, 'sigma': 1.5},
            'high': {'edge_threshold': 80, 'density_threshold': 0.1, 'sigma': 1.0}
        }
        def detect(self, image_np, sensitivity='medium'):
            """
            Detects busy areas in the image using edge detection and gradient analysis.
            
            Args:
                image_np: numpy array of image
                sensitivity: 'low', 'medium', or 'high'
                
            Returns:
                List of dictionaries with busy area coordinates
            """
            # Get parameters based on sensitivity
            params = self.sensitivity_presets.get(sensitivity, self.sensitivity_presets['medium'])
            
            # Convert to grayscale if it's not already
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
                
            # 1. Edge detection
            edges = cv2.Canny(gray, params['edge_threshold'], params['edge_threshold'] * 2)
            
            # 2. Texture complexity analysis (using gradients)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Normalize gradient magnitude
            if gradient_magnitude.max() > 0:
                gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
            
            # 3. Combine edge and gradient information
            complexity_map = edges.astype(float) / 255 + gradient_magnitude
            complexity_map = np.clip(complexity_map, 0, 1)
            
            # 4. Apply Gaussian smoothing to the map
            complexity_map = gaussian_filter(complexity_map, sigma=params['sigma'])
            
            # 5. Segment the image into grid cells and analyze each cell
            h, w = complexity_map.shape
            cell_size = min(h, w) // 10  # Divide image into approximately 10x10 grid
            cell_size = max(cell_size, 20)  # Minimum cell size of 20px
            
            busy_areas = []
            
            # Iterate through grid cells
            for y in range(0, h - cell_size + 1, cell_size):
                for x in range(0, w - cell_size + 1, cell_size):
                    cell = complexity_map[y:y+cell_size, x:x+cell_size]
                    
                    # Calculate average complexity in this cell
                    avg_complexity = np.mean(cell)
                    
                    # If complexity is above threshold, mark as busy area
                    if avg_complexity > params['density_threshold']:
                        busy_areas.append({
                            'x': int(x),
                            'y': int(y),
                            'width': int(cell_size),
                            'height': int(cell_size),
                            'complexity': float(avg_complexity)
                        })
            
            # 6. Merge adjacent busy areas
            merged_areas = self._merge_adjacent_areas(busy_areas, cell_size)
            
            # Sort by complexity (highest first)
            merged_areas.sort(key=lambda area: area['complexity'], reverse=True)
            
            return merged_areas    
            
        def _merge_adjacent_areas(self, areas, cell_size):
            """Merge adjacent busy areas to form larger regions."""
            if not areas:
                return []
                
            # Helper function to check if two areas overlap or are adjacent
            def are_adjacent(a1, a2, tolerance=1.5):
                # Expand the first area slightly to detect adjacency
                expanded_a1 = {
                    'x': a1['x'] - cell_size/tolerance,
                    'y': a1['y'] - cell_size/tolerance,
                    'width': a1['width'] + cell_size/tolerance*2,
                    'height': a1['height'] + cell_size/tolerance*2
                }
                
                # Check if a2 intersects with the expanded a1
                return not (expanded_a1['x'] + expanded_a1['width'] < a2['x'] or
                        a2['x'] + a2['width'] < expanded_a1['x'] or
                        expanded_a1['y'] + expanded_a1['height'] < a2['y'] or
                        a2['y'] + a2['height'] < expanded_a1['y'])
                