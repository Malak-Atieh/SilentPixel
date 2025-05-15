const express = require('express');
const crypto = require('crypto');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const upload = multer();

// AES-256 Decryption Helper - Fixed to handle IV errors
function decryptMessage(encrypted, password) {
  try {
    // Split the encrypted string into IV and data parts
    const parts = encrypted.split(':');
    if (parts.length !== 2) {
      throw new Error('Invalid encrypted format - expected IV:encryptedData');
    }
    
    const [ivBase64, encryptedData] = parts;
    
    // Decode IV from base64 and check its length
    let iv;
    try {
      iv = Buffer.from(ivBase64, 'base64');
      if (iv.length !== 16) {
        throw new Error(`IV length must be 16 bytes, got ${iv.length}`);
      }
    } catch (ivError) {
      throw new Error(`Invalid IV: ${ivError.message}`);
    }
    
    // Generate key from password
    const key = crypto.scryptSync(password, 'salt', 32);
    
    // Create decipher and decrypt
    const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
    
    // Decrypt the message
    let decrypted;
    try {
      decrypted = decipher.update(encryptedData, 'base64', 'utf8');
      decrypted += decipher.final('utf8');
    } catch (decryptError) {
      throw new Error(`Decryption operation failed: ${decryptError.message}`);
    }
    
    return decrypted;
  } catch (err) {
    console.error('Decryption error details:', err);
    throw new Error(`Decryption failed: ${err.message}`);
  }
}

router.post('/', upload.single('image'), async (req, res) => {
  try {
    const { key, password } = req.body;
    
    if (!key || !req.file) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    console.log(`Extracting with key: ${key}`);
    
    // Prepare data for ML service
    const form = new FormData();
    form.append('image', req.file.buffer, { filename: 'img.png' });
    form.append('key', key);

    // Forward to ML decode service
    let mlResponse;
    try {
      mlResponse = await axios.post('http://localhost:5003/extract', form, {
        headers: form.getHeaders(),
        maxContentLength: Infinity,
        maxBodyLength: Infinity
      });
    } catch (mlError) {
      console.error('ML service error:', mlError.message);
      return res.status(500).json({
        error: 'ML extraction service failed',
        message: mlError.message
      });
    }

    console.log('Extracted raw message length:', mlResponse.data.message.length);
    
    let extractedMessage = mlResponse.data.message;
    
    // For debugging, log part of the message
    console.log('Message starts with:', extractedMessage.substring(0, 50));
    console.log('Message format check - contains colon:', extractedMessage.includes(':'));
    
    // Attempt decryption if password is provided
    if (password && extractedMessage.includes(':')) {
      try {
        extractedMessage = decryptMessage(extractedMessage, password);
        console.log('Decryption successful');
      } catch (decryptErr) {
        console.error('Decryption error:', decryptErr);
        return res.status(403).json({ 
          error: 'Decryption failed', 
          message: decryptErr.message 
        });
      }
    } else if (password) {
      console.warn('Password provided but message format is not encrypted (missing colon separator)');
    }

    res.json({ 
      message: extractedMessage,
      extracted_length: mlResponse.data.message.length
    });
  } catch (err) {
    console.error('General error in decode route:', err);
    res.status(500).json({ 
      error: 'Extraction failed', 
      message: err.message || 'Unknown error' 
    });
  }
});

module.exports = router;