const express = require('express');
const crypto = require('crypto');
const multer = require('multer');
const axios = require('axios');
const QRCode = require('qrcode');
const FormData = require('form-data');

const router = express.Router();
const upload = multer();

// AES-256 Encryption Helper - Fixed to ensure proper IV format
function encryptMessage(message, password) {
  try {
    // Generate key from password
    const key = crypto.scryptSync(password, 'salt', 32);
    
    // Generate a random 16-byte IV
    const iv = crypto.randomBytes(16);
    
    // Create cipher and encrypt
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    let encrypted = cipher.update(message, 'utf8', 'base64');
    encrypted += cipher.final('base64');
    
    // Return IV and encrypted data properly formatted
    return `${iv.toString('base64')}:${encrypted}`;
  } catch (err) {
    console.error('Encryption error:', err);
    throw new Error(`Encryption failed: ${err.message}`);
  }
}

router.post('/', upload.single('image'), async (req, res) => {
  try {
    const { message, password } = req.body;
    
    if (!message || !req.file) {
      return res.status(400).json({ error: 'Missing required fields' });
    }
    
    console.log('Input message length:', message.length);
    
    // Encrypt message if password is provided
    let encrypted;
    let encryptionUsed = false;
    
    if (password) {
      try {
        encrypted = encryptMessage(message, password);
        encryptionUsed = true;
        console.log('Encryption successful, encrypted length:', encrypted.length);
        
        // Validate encrypted format
        if (!encrypted.includes(':')) {
          throw new Error('Invalid encryption format - missing separator');
        }
      } catch (encryptErr) {
        console.error('Encryption error:', encryptErr);
        return res.status(500).json({
          error: 'Encryption failed',
          message: encryptErr.message
        });
      }
    } else {
      encrypted = message;
      console.log('No encryption used');
    }

    // Check if encrypted message is too large for image
    const maxMessageSize = req.file.size / 10; // Rough estimate - about 10% of image size
    if (encrypted.length > maxMessageSize) {
      return res.status(400).json({
        error: 'Message too large',
        message: `Message size (${encrypted.length}) exceeds capacity for this image (~${Math.floor(maxMessageSize)})`
      });
    }

    // Generate a proper steganography key
    const stegKey = 'STG#' + crypto.randomBytes(6).toString('hex').toUpperCase();
    console.log(`Generated steganography key: ${stegKey}`);
    
    // Create metadata for QR code
    const metadata = {
      alg: 'LSB',
      key_hint: stegKey.slice(-6),
      encrypted: encryptionUsed,
      length: encrypted.length,
      timestamp: Date.now()
    };

    // Generate QR code with metadata
    const qrDataUrl = await QRCode.toDataURL(JSON.stringify(metadata));
    console.log('QR code generated with metadata');

    // Prepare form data to send to embedding service
    const form = new FormData();
    form.append('image', req.file.buffer, { filename: 'img.png' });
    form.append('message', encrypted);
    form.append('key', stegKey);

    // Send to ML service
    console.log('Sending to embedding service...');
    const mlResponse = await axios.post('http://localhost:5002/embed', form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    console.log('Embedding successful');

    // Return results to client
    res.json({
      stego_image_base64: mlResponse.data.stego_image_base64,
      key: stegKey,
      qr_code_base64: qrDataUrl.split(',')[1],
      metadata: metadata
    });
  } catch (err) {
    console.error('General error in encode route:', err);
    res.status(500).json({ 
      error: 'Encode failed', 
      message: err.message || 'Unknown error'
    });
  }
});

module.exports = router;