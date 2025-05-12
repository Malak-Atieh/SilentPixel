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

