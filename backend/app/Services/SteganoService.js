const crypto = require('crypto');
const sharp = require('sharp');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs').promises;
class steganoService {

  static   async embedMessage(imageBuffer, message, password, busyAreas = []) {
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

  static async decodeMessage(imageFile, password ) {
    try{
      const decode = new LSBDecoder();
      const result = await decode.decode(imageFile, password);

      return {
        message: result.message,
        watermark: result.watermark || null,
        qrCodeData: result.qrCodeData || null,
      }
    } catch (error) {
      throw new Error(`Decoding failed: ${error.message}`);
    }
  }

  async detectSteganography(imagePath) {
    try{
      const detectionResult = this.MLService.detectHiddenData(imagePath);
      return detectionResult;
    }catch (error) {
      throw new Error(`Steganography detection failed: ${error.message}`);
    }
  }
}

module.exports = steganoService;