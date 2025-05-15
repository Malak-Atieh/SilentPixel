# ml_services/analyze_service.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    img_bytes = await image.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(image)

    edges = cv2.Canny(np_img, 50, 150)
    heatmap = cv2.applyColorMap(edges, cv2.COLORMAP_JET)

    # Encode as base64 PNG
    _, buffer = cv2.imencode('.png', heatmap)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"heatmap_base64": encoded}
