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