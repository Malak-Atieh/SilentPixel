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
      const maxCapacity = Math.floor(pixels.length / 4);
      if (dataToHide.length > maxCapacity) {
        throw new Error(`Message too large. Max capacity: ${Math.floor(maxCapacity / 8)} bytes`);
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
}

module.exports = steganoService;