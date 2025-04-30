from flask import Flask, request, send_file
import cv2
import numpy as np
from io import BytesIO
from stegano import lsb

app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode():
    if 'image' not in request.files or not request.form.get('message'):
        return {"error": "Image and message required"}, 400
    
    # 1. Process image
    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Hide message (using LSB from stegano)
    secret = lsb.hide(img, request.form['message'])
    
    # 3. Return encoded image
    _, encoded_img = cv2.imencode('.png', secret)
    return send_file(
        BytesIO(encoded_img.tobytes()),
        mimetype='image/png'
    )

@app.route('/decode', methods=['POST'])
def decode():
    if 'image' not in request.files:
        return {"error": "Image required"}, 400
    
    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    message = lsb.reveal(img)
    return {"message": message}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)