from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import hashlib

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

def generate_positions(key, total_pixels, message_bits):
    """Generate pixel positions based on key for embedding"""
    # Create a deterministic seed from the key
    seed_str = hashlib.sha256(key.encode()).hexdigest()
    seed = int(seed_str, 16) % 10**8
    
    print(f"Embedding using key: {key}")
    print(f"Generated seed: {seed} from hash: {seed_str[:16]}...")
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Generate unique positions
    positions = np.random.permutation(total_pixels)[:message_bits]
    positions.sort()  # Sort for sequential access
    
    print(f"Generated {len(positions)} positions for embedding")
    return positions

def embed_lsb(image_array, message, key):
    # Add delimiter for extraction
    full_message = message + '|||END|||'
    
    # Convert message to binary
    binary = ''.join(format(ord(char), '08b') for char in full_message)
    message_bits = len(binary)
    
    print(f"Message length: {len(message)}, with delimiter: {len(full_message)}")
    print(f"Binary message length: {message_bits} bits")
    
    # Get image dimensions and calculate total pixels
    flat = image_array.flatten()
    total_pixels = len(flat)
    
    print(f"Image dimensions: {image_array.shape}, total pixels: {total_pixels}")
    
    # Ensure image has enough capacity
    if message_bits > total_pixels:
        raise ValueError(f"Message too large for this image. Needs {message_bits} pixels, image has {total_pixels}")
        
    # Generate bit positions using the key
    positions = generate_positions(key, total_pixels, message_bits)
    
    # Make a copy of the image array to avoid modifying the original
    result = image_array.copy()
    result_flat = result.flatten()
    
    # Embed each bit of the message
    for i, bit in enumerate(binary):
        if i < len(positions):
            pos = positions[i]
            # Clear the LSB and set it to the message bit
            result_flat[pos] = (result_flat[pos] & ~1) | int(bit)
    
    # Reshape back to original dimensions
    return result_flat.reshape(image_array.shape)

@app.post("/embed")
async def embed(image: UploadFile = File(...), message: str = Form(...), key: str = Form(...)):
    try:
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        np_img = np.array(pil_image)
        
        print(f"Processing image: {pil_image.size}")

        stego_img = embed_lsb(np_img, message, key)
        stego_pil = Image.fromarray(stego_img.astype(np.uint8))

        # Save the image to a buffer
        buffer = BytesIO()
        stego_pil.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"Embedding complete. Output image size: {len(encoded)}")

        # Return additional metadata for debugging
        return {
            "stego_image_base64": encoded,
            "message_length": len(message),
            "bits_embedded": len(message) * 8 + 8 * 8  # Message + delimiter
        }
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")