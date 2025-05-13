const BinaryConverter = require('./binaryConverter');
const PixelSelector = require('./pixelSelector');
const EncryptionService = require('../../Services/EncryptionService');
const Jimp = require('jimp');

class SteganoCore {
  static async embed(imageBuffer, message, password, busyAreas = []) {
    const image = await Jimp.read(imageBuffer);
    const { width, height, data } = image.bitmap;

    const encryptedMsg = EncryptionService.encrypt(message, password);
    const binaryMsg = BinaryConverter.textToBinary(encryptedMsg);
    const header = BinaryConverter.numberToBinary(binaryMsg.length, 32);
    const dataToHide = header + binaryMsg;
    
    const pixelIndices = PixelSelector.getIndices(
      width, 
      height, 
      busyAreas, 
      dataToHide.length
    );
    
    this._embedData(imageData.data, pixelIndices, dataToHide);
    
    return await image.getBufferAsync(Jimp.MIME_PNG);
  }

  static async extract(imageBuffer, password) {
     const image = await Jimp.read(imageBuffer);
    const { data } = image.bitmap;
    
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