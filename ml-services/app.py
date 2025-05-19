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

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MAX_CONTENT_LENGTH = 16 * 1024 * 1024  
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

class SteganographyCNN(nn.Module):
    """Deep learning model for steganography detection."""
    
    def __init__(self):
        super(SteganographyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3) 
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def detect(self, image):
 
        self.eval()
        with torch.no_grad():
            output = self.forward(image)
            probabilities = torch.softmax(output, dim=1)[0]
                    
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[pred_class].item() * 100
                
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
    def __init__(self):
        # Sensitivity presets (adjust based on empirical testing)
        self.sensitivity_presets = {
            'low': {'edge_threshold': 120, 'density_threshold': 0.3, 'sigma': 2.0},
            'medium': {'edge_threshold': 100, 'density_threshold': 0.2, 'sigma': 1.5},
            'high': {'edge_threshold': 80, 'density_threshold': 0.1, 'sigma': 1.0}
        }
    def detect(self, image_np, sensitivity='medium'):
        params = self.sensitivity_presets.get(sensitivity, self.sensitivity_presets['medium'])
            
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
                
        edges = cv2.Canny(gray, params['edge_threshold'], params['edge_threshold'] * 2)
            
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
            
        complexity_map = edges.astype(float) / 255 + gradient_magnitude
        complexity_map = np.clip(complexity_map, 0, 1)
            
        complexity_map = gaussian_filter(complexity_map, sigma=params['sigma'])
            
        h, w = complexity_map.shape
        cell_size = min(h, w) // 10  
        cell_size = max(cell_size, 20)
            
        busy_areas = []
            
        for y in range(0, h - cell_size + 1, cell_size):
            for x in range(0, w - cell_size + 1, cell_size):
                cell = complexity_map[y:y+cell_size, x:x+cell_size]
                    
                avg_complexity = np.mean(cell)
                    
                if avg_complexity > params['density_threshold']:
                    busy_areas.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(cell_size),
                        'height': int(cell_size),
                        'complexity': float(avg_complexity)
                    })
            
        merged_areas = self._merge_adjacent_areas(busy_areas, cell_size)
            
        merged_areas.sort(key=lambda area: area['complexity'], reverse=True)
            
        return merged_areas    
            
    def _merge_adjacent_areas(self, areas, cell_size):
        """Merge adjacent busy areas to form larger regions."""
        if not areas:
            return []
                
        def are_adjacent(a1, a2, tolerance=1.5):
            expanded_a1 = {
                'x': a1['x'] - cell_size/tolerance,
                'y': a1['y'] - cell_size/tolerance,
                'width': a1['width'] + cell_size/tolerance*2,
                'height': a1['height'] + cell_size/tolerance*2
            }
                
            return not (expanded_a1['x'] + expanded_a1['width'] < a2['x'] or
                    a2['x'] + a2['width'] < expanded_a1['x'] or
                    expanded_a1['y'] + expanded_a1['height'] < a2['y'] or
                    a2['y'] + a2['height'] < expanded_a1['y'])
            
        def merge(a1, a2):
                x1 = min(a1['x'], a2['x'])
                y1 = min(a1['y'], a2['y'])
                x2 = max(a1['x'] + a1['width'], a2['x'] + a2['width'])
                y2 = max(a1['y'] + a1['height'], a2['y'] + a2['height'])
                
                avg_complexity = (a1['complexity'] * (a1['width'] * a1['height']) + 
                                a2['complexity'] * (a2['width'] * a2['height'])) / \
                                ((x2 - x1) * (y2 - y1))
                
                return {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'complexity': avg_complexity
                }
            
        merged = list(areas)
        while True:
            merged_this_round = False
                
            for i in range(len(merged)):
                if merged[i] is None:
                    continue
                    
                for j in range(i+1, len(merged)):
                    if merged[j] is None:
                        continue
                        
                    if are_adjacent(merged[i], merged[j]):
                        merged[i] = merge(merged[i], merged[j])
                        merged[j] = None  
                        merged_this_round = True
                
            merged = [area for area in merged if area is not None]
                
            if not merged_this_round:
                break
            
        return merged    
        

steg_model = None
busy_area_detector = None

def load_steganography_model():
    global steg_model
    
    if steg_model is None:
        logger.info("Loading steganography detection model...")
        model = SteganographyCNN()
        
        model_path = os.environ.get('STEG_MODEL_PATH', 'models/steg_model.pth')
        try:
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                logger.info("Loaded steganography model from %s", model_path)
            else:
                logger.warning("Model file not found at %s. Using untrained model.", model_path)
        except Exception as e:
            logger.error("Failed to load steganography model: %s", str(e))
        
        model.eval()
        steg_model = model
    
    return steg_model

def load_busy_area_detector():
    global busy_area_detector
    
    if busy_area_detector is None:
        logger.info("Initializing busy area detector...")
        busy_area_detector = BusyAreaDetector()
    
    return busy_area_detector

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):

    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_np = np.array(img)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  
    
    return img_tensor, img_np


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "Service is running"})


@app.route('/api/analyze', methods=['POST'])
def analyze_image():

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty file provided"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        image_bytes = file.read()
        img_tensor, _ = preprocess_image(image_bytes)
        
        model = load_steganography_model()
        
        results = model.detect(img_tensor)
        
        results["imageSize"] = len(image_bytes)
        results["fileName"] = file.filename
        
        return jsonify(results)
        
    except Exception as e:
        logger.error("Error analyzing image: %s", str(e), exc_info=True)
        return jsonify({"error": "Failed to analyze image", "message": str(e)}), 500


@app.route('/api/detect-busy-areas', methods=['POST'])
def detect_busy_areas():

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty file provided"}), 400
            
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        sensitivity = request.form.get('sensitivity', 'medium').lower()
        if sensitivity not in ['low', 'medium', 'high']:
            sensitivity = 'medium'
        
        image_bytes = file.read()
        _, img_np = preprocess_image(image_bytes)
        
        detector = load_busy_area_detector()
        
        busy_areas = detector.detect(img_np, sensitivity)
        
        return jsonify({
            "busyAreas": busy_areas,
            "imageSize": {
                "width": img_np.shape[1],
                "height": img_np.shape[0]
            },
            "sensitivity": sensitivity
        })
        
    except Exception as e:
        logger.error("Error detecting busy areas: %s", str(e), exc_info=True)
        return jsonify({"error": "Failed to detect busy areas", "message": str(e)}), 500


if __name__ == '__main__':
    load_steganography_model()
    load_busy_area_detector()
    
    port = int(os.environ.get('PORT', 5001))
    
    logger.info(f"Starting ML microservice on port {port}")
    serve(app, host='0.0.0.0', port=port)