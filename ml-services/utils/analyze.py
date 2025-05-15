# utils/analyze.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

def load_model(path='models/steg_detector.pth'):
    model = torch.load(path, map_location='cpu')
    model.eval()
    return model

def predict_image(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1).squeeze()
    class_idx = torch.argmax(prob).item()
    confidence = prob[class_idx].item()
    class_map = {0: 'clean', 1: 'lsb', 2: 'dct'}
    return {
        'hasSteganography': class_idx != 0,
        'method': class_map[class_idx],
        'confidence': round(confidence, 4)
    }
