const BinaryConverter = require('./binaryConverter');
const PixelSelector = require('./pixelSelector');
const EncryptionService = require('../../Services/encryptionService');
const ImageProcessor = require('./imageProcessor');

class SteganoCore {
  static async embed(imageBuffer, message, password, busyAreas = []) {
    const { canvas, ctx } = await ImageProcessor.loadImageToCanvas(imageBuffer);
    const imageData = ImageProcessor.getImageData(ctx, canvas.width, canvas.height);
    
    const encryptedMsg = EncryptionService.encrypt(message, password);
    const binaryMsg = BinaryConverter.textToBinary(encryptedMsg);
    const header = BinaryConverter.numberToBinary(binaryMsg.length, 32);
    const dataToHide = header + binaryMsg;
    
    const pixelIndices = PixelSelector.getIndices(
      canvas.width, 
      canvas.height, 
      busyAreas, 
      dataToHide.length
    );
    
    this._embedData(imageData.data, pixelIndices, dataToHide);
    
    ImageProcessor.updateImageData(ctx, imageData);
    return ImageProcessor.canvasToBuffer(canvas);
  }
  static async extract(imageBuffer, password) {
    const { canvas, ctx } = await ImageProcessor.loadImageToCanvas(imageBuffer);
    const imageData = ImageProcessor.getImageData(ctx, canvas.width, canvas.height);
    
    const binaryHeader = this._extractBits(imageData.data, 0, 32);
    const messageLength = parseInt(binaryHeader, 2);
    
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
    for (let i = 0; i < length; i++) {
      const pixelIndex = (startIndex + i) * 4;
      bits += (pixels[pixelIndex + 2] & 1).toString(); // Using blue channel
    }
    return bits;
  }
}

module.exports = SteganoCore;