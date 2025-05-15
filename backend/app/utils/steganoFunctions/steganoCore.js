const BinaryConverter = require('./binaryConverter');
const PixelSelector = require('./pixelSelector');
const EncryptionService = require('../../Services/EncryptionService');
const ImageProcessor = require('../imageProcessor');

class SteganoCore {
  static async embed(imageBuffer, message, password, busyAreas = []) {
      // Load the image using Sharp
    const { image, metadata } = await ImageProcessor.loadImage(imageBuffer);
      
      // Get raw pixel data
    const imageData = await ImageProcessor.getImageData(image);
      
    const encryptedMsg = EncryptionService.encrypt(message, password  || '');
    const binaryMsg = BinaryConverter.textToBinary(encryptedMsg);
    const header = BinaryConverter.numberToBinary(binaryMsg.length, 32);
    const dataToHide = header + binaryMsg;
    
    const pixelIndices = PixelSelector.getIndices(
      imageData.width,
      imageData.height,
      busyAreas, 
      dataToHide.length
    );
    
      // Validate we have enough capacity
      if (pixelIndices.length < dataToHide.length) {
        throw new Error(`Insufficient capacity: need ${dataToHide.length} pixels, but only ${pixelIndices.length} available.`);
      }

    this._embedData(imageData.data, pixelIndices, dataToHide);
    
          // Update the image with modified pixel data
      const updatedImage = ImageProcessor.updateImage(imageData);
      
    return await ImageProcessor.imageToBuffer({ image: updatedImage });
  }

  static async extract(imageBuffer, password) {
       // Load the image using Sharp
      const { image } = await ImageProcessor.loadImage(imageBuffer);
      
      // Get raw pixel data
      const imageData = await ImageProcessor.getImageData(image);
      
    const binaryHeader = this._extractBits(imageData.data, 0, 32);
    const messageLength = parseInt(binaryHeader, 2);
    
          if (messageLength <= 0 || messageLength > imageData.width * imageData.height * 3) {
        throw new Error('Invalid message length detected. This may not be a steganographic image.');
      }
      
    const binaryMsg = this._extractBits(imageData.data, 32, messageLength);
    const encryptedMsg = BinaryConverter.binaryToText(binaryMsg);
    
    return EncryptionService.decrypt(encryptedMsg, password);
  }

  static _embedData(pixels, pixelIndices, binaryData) {
    const channels = [0, 1, 2]; // R, G, B channels
    
    for (let i = 0; i < binaryData.length; i++) {
      const pixelIndex = pixelIndices[i] * 4;
      const channel = channels[Math.floor(Math.random() * 3)];
      const bit = parseInt(binaryData[i]);
      pixels[pixelIndex + channel] = (pixels[pixelIndex + channel] & 0xFE) | bit;
    }
  }

  static _extractBits(pixels, startIndex, length) {
    let bits = '';
    const channels = [0, 1, 2]; 
    for (let i = 0; i < length; i++) {
      const pixelIndex = (startIndex + i) * 4;
      for (const channel of channels) {
        bits += (pixels[pixelIndex + channel] & 1).toString();
      }
    }
    return bits;
  }
}

module.exports = SteganoCore;