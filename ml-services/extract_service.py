from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from io import BytesIO
import hashlib
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

def generate_positions(key, total_pixels, max_bits):
    """Generate pixel positions based on key for extraction"""
    # Create a deterministic seed from the key
    seed_str = hashlib.sha256(key.encode()).hexdigest()
    seed = int(seed_str, 16) % 10**8
    
    print(f"Extraction using key: {key}")
    print(f"Generated seed: {seed} from hash: {seed_str[:16]}...")
    
    # Set the seed for reproducibility
    np.random.seed(seed)
    
    # Generate unique positions
    positions = np.random.permutation(total_pixels)[:max_bits]
    positions.sort()  # Sort for sequential access
    
    print(f"Generated {len(positions)} positions for extraction")
    return positions

def extract_lsb(image_array, key):
    flat = image_array.flatten()
    total_pixels = len(flat)
    
    print(f"Image shape: {image_array.shape}, total pixels: {total_pixels}")
    
    # We don't know message length, so use a reasonable maximum
    # Each character is 8 bits, plus end marker
    max_bits = min(total_pixels, 500000)  # Will extract up to ~62KB of text
    
    # Generate bit positions using the key - same algorithm as embedding
    positions = generate_positions(key, total_pixels, max_bits)
    
    # Extract bits from the positions
    bits = []
    for pos in positions:
        if pos < len(flat):
            bits.append(str(flat[pos] & 1))
    
    print(f"Extracted {len(bits)} bits")
    
    # Convert bits to characters
    chars = []
    for i in range(0, len(bits), 8):
        if i + 8 <= len(bits):  # Ensure we have 8 bits
            byte = bits[i:i+8]
            byte_val = int(''.join(byte), 2)
            try:
                c = chr(byte_val)
                chars.append(c)
                
                # Check for end marker
                if len(chars) >= 8:
                    recent = ''.join(chars[-8:])
                    if recent == '|||END|||':
                        print(f"Found end marker at position {len(chars)}")
                        break
            except ValueError:
                print(f"Invalid byte value: {byte_val}")
    
    # Remove end marker
    message = ''.join(chars)
    if '|||END|||' in message:
        message = message.split('|||END|||')[0]
    
    print(f"Extracted message length: {len(message)}")
    # Print first 50 chars for debugging
    if len(message) > 0:
        print(f"Message starts with: {message[:50]}")
    
    return message

@app.post("/extract")
async def extract(image: UploadFile = File(...), key: str = Form(...)):
    try:
        image_data = await image.read()
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        np_img = np.array(pil_image)
        
        print(f"Processing image: {pil_image.size}")

        message = extract_lsb(np_img, key)
        
        return {"message": message}
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")