const crypto = require('crypto');
const sharp = require('sharp');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs').promises;
class steganoService {

  static async embedMessage(imageBuffer, message, password, busyAreas = []) {
    try {
      // Load the image
      const image = await loadImage(imageBuffer);

      // Create canvas with same dimensions
      const canvas = createCanvas(image.width, image.height);
      const ctx = canvas.getContext('2d');
      
      // Draw image on canvas
      ctx.drawImage(image, 0, 0);
      
      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixels = imageData.data;
      
      // Encrypt the message with the password
      const encryptedMsg = this.encryptMessage(message, password);
      
      // Convert message to binary
      const binaryMsg = this._textToBinary(encryptedMsg);
      
      // Add header with message length
      const header = this._numberToBinary(binaryMsg.length, 32);
      const dataToHide = header + binaryMsg;
      
      // Check if image has enough capacity
      const requiredCapacity = dataToHide.length;
      const availablePixels = this._getPixelIndices(canvas.width, canvas.height, busyAreas, requiredCapacity);
      if (availablePixels.length < requiredCapacity) {
        throw new Error(`Insufficient capacity: need ${requiredCapacity} pixels, but only ${availablePixels.length} available.`);
      }

      // Determine pixel indices to modify based on busy areas
      const pixelIndices = this._getPixelIndices(canvas.width, canvas.height, busyAreas, dataToHide.length);      
      
      // Embed data
      for (let i = 0; i < dataToHide.length; i++) {
        const pixelIndex = pixelIndices[i] * 4; // Each pixel has 4 values (RGBA)
        
        // Only modify the least significant bit of the blue channel (less noticeable)
        const bit = parseInt(dataToHide[i]);
        pixels[pixelIndex + 2] = (pixels[pixelIndex + 2] & 0xFE) | bit;
      }
      
      // Update canvas with modified pixels
      ctx.putImageData(imageData, 0, 0);
      
      // Convert canvas to buffer
      const modifiedBuffer = canvas.toBuffer('image/png');
      
      return modifiedBuffer;
    } catch (error) {
      throw new Error(`Encoding failed: ${error.message}`);
    }   

  }

  static async extractMessage(imageBuffer, password ) {
    try{
      // Load the image
      const image = await loadImage(imageBuffer);
            
      // Create canvas
      const canvas = createCanvas(image.width, image.height);
      const ctx = canvas.getContext('2d');
      
      // Draw image on canvas
      ctx.drawImage(image, 0, 0);
      
      // Get image data
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixels = imageData.data;
      
      // Extract header (first 32 bits)
      let binaryHeader = '';
      for (let i = 0; i < 32; i++) {
        const pixelIndex = i * 4;
        binaryHeader += (pixels[pixelIndex + 2] & 1).toString();
      }
      
      // Get message length from header
      const messageLength = parseInt(binaryHeader, 2);
      if (isNaN(messageLength) || messageLength > width * height) {
        throw new Error('Invalid or corrupted header detected.');
      }
      
      // Extract binary message
      let binaryMsg = '';
      for (let i = 0; i < messageLength; i++) {
        const pixelIndex = (i + 32) * 4;
        binaryMsg += (pixels[pixelIndex + 2] & 1).toString();
      }
      
      // Convert binary to text
      const encryptedMsg = this._binaryToText(binaryMsg);
      
      // Decrypt the message
      const message = this.decryptMessage(encryptedMsg, password);
      
      return message;
    } catch (error) {
      throw new Error(`Decoding failed: ${error.message}`);
    }
  }

  async extractMessageWithQR(imageBuffer, password, qrData) {
    // Extract message normally
    const message = await this.extractMessage(imageBuffer, password);
    
    // Verify message hash from QR code
    const calculatedHash = this.generateMessageHash(message, password);
    
    if (calculatedHash !== qrData.messageHash) {
      throw new Error('Message integrity check failed. The image may have been tampered with.');
    }
    
    return message;
  }
  
  encryptMessage(message, password) {
    // Generate a key and IV from the password
    const key = crypto.scryptSync(password, 'salt', 32);
    const iv = crypto.randomBytes(16);
    
    // Create cipher
    const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
    
    // Encrypt message
    let encrypted = cipher.update(message, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    // Combine IV and encrypted message
    return iv.toString('hex') + ':' + encrypted;
  }

  decryptMessage(encryptedMsg, password) {
    try {
      // Split IV and encrypted message
      const parts = encryptedMsg.split(':');
      const iv = Buffer.from(parts[0], 'hex');
      const encryptedText = parts[1];
      
      // Generate key from password
      const key = crypto.scryptSync(password, 'salt', 32);
      
      // Create decipher
      const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
      
      // Decrypt message
      let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      
      return decrypted;
    } catch (error) {
      throw new Error('Decryption failed. Check if the password is correct.');
    }
  }
  
  generateMessageHash(message, password) {
    return crypto.createHash('sha256')
      .update(message + password)
      .digest('hex');
  }
  
  _getPixelIndices(width, height, busyAreas, dataLength) {
    const indices = [];
    
    // If we have busy areas, prioritize those pixels
    if (busyAreas && busyAreas.length > 0) {
      // Create a map of all busy area pixels
      const busyPixels = new Set();
      
      for (const area of busyAreas) {
        for (let y = area.y; y < area.y + area.height; y++) {
          for (let x = area.x; x < area.x + area.width; x++) {
            if (x < width && y < height) {
              const index = y * width + x;
              busyPixels.add(index);
            }
          }
        }
      }
      
      // Convert set to array and shuffle for better distribution
      const busyIndices = Array.from(busyPixels);
      this._shuffleArray(busyIndices);
      
      // Use busy pixels first
      for (let i = 0; i < Math.min(dataLength, busyIndices.length); i++) {
        indices.push(busyIndices[i]);
      }
      
      // If we need more pixels, use sequential pixels for the rest
      if (dataLength > busyIndices.length) {
        let currentIndex = 0;
        while (indices.length < dataLength && currentIndex < width * height) {
          if (!busyPixels.has(currentIndex)) {
            indices.push(currentIndex);
          }
          currentIndex++;
        }
        if (indices.length < dataLength) {
          throw new Error('Not enough non-busy pixels to embed full message.');
        }
      }
    } else {
      // No busy areas defined, use sequential pixels
      for (let i = 0; i < dataLength; i++) {
        indices.push(i);
      }
    }
    
    return indices;
  }

  _textToBinary(text) {
    let binary = '';
    for (let i = 0; i < text.length; i++) {
      const charCode = text.charCodeAt(i);
      const binChar = charCode.toString(2).padStart(8, '0');
      binary += binChar;
    }
    return binary;
  }

  _binaryToText(binary) {
    let text = '';
    for (let i = 0; i < binary.length; i += 8) {
      const byte = binary.substr(i, 8);
      const charCode = parseInt(byte, 2);
      text += String.fromCharCode(charCode);
    }
    return text;
  }

  _numberToBinary(num, length) {
    return num.toString(2).padStart(length, '0');
  }

  _shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
}

module.exports = steganoService;