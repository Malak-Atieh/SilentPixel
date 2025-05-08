const BinaryConverter = require('./binaryConverter');
const PixelSelector = require('./pixelSelector');
const EncryptionService = require('../../Services/encryptionService');
const ImageProcessor = require('./imageProcessor');

class SteganographyCore {
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

}

module.exports = SteganographyCore;