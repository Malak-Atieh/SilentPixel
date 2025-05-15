const BinaryConverter = require('./binaryConverter');
const PixelSelector = require('./pixelSelector');
const EncryptionService = require('../../Services/EncryptionService');
const ImageProcessor = require('../imageProcessor');

class SteganoCore {
  static SIGNATURE = "11010010";

  static async embed(imageBuffer, message, password, busyAreas = [], protectedZones = []) {
      if (!message || message.length === 0) {
        throw new Error('Message cannot be empty');
      }
        
      if (password && typeof password !== 'string') {
        throw new Error('Password must be a string');
      }
    const { image } = await ImageProcessor.loadImage(imageBuffer);
      
    const imageData = await ImageProcessor.getImageData(image);
      
    const encryptedMsg = EncryptionService.encrypt(message, password);
    const binaryMsg = BinaryConverter.textToBinary(encryptedMsg);
    // Create header: signature(8) + length(32) + is_encrypted(1)
    const header = this.SIGNATURE + 
                  BinaryConverter.numberToBinary(binaryMsg.length, 32) + 
                  (password ? "1" : "0");
        const dataToHide = header + binaryMsg;

    const pixelsNeeded = Math.ceil(dataToHide.length / 3);
    const totalPixels = imageData.width * imageData.height;
    
    const pixelIndices = Array.from({length: Math.min(pixelsNeeded, totalPixels)}, (_, i) => i);
      if (pixelIndices.length < pixelsNeeded) {
        throw new Error(`Insufficient capacity: need ${dataToHide.length} pixels, but only ${pixelIndices.length} available.`);
      }

    this._embedData(
      imageData.data, 
      pixelIndices, 
      dataToHide,
      imageData.width,
      protectedZones
    );
   
      const updatedImage = ImageProcessor.updateImage(imageData);
      
    return await ImageProcessor.imageToBuffer({ image: updatedImage });
  }

  static async extract(imageBuffer, password) {
     
      const { image } = await ImageProcessor.loadImage(imageBuffer);
      
      const imageData = await ImageProcessor.getImageData(image);
      const signature = this._extractBitsAt(imageData.data, 0, 8);

         if (signature !== this.SIGNATURE) {
      throw new Error('Invalid signature. This image does not appear to contain hidden data.');
    }
    
    const binaryLength = this._extractBitsAt(imageData.data, 8, 32);
    const messageLength = parseInt(binaryLength, 2);

    const maxPossibleLength = imageData.width * imageData.height * 3 - this.HEADER_SIZE;
    if (messageLength <= 0 || messageLength > maxPossibleLength) {
      throw new Error('Invalid message length detected.');
    }
    const isEncrypted = this._extractBitsAt(imageData.data, 40, 1) === "1";
  

    const binaryMsg = this._extractBitsAt(imageData.data, 41, messageLength);
    const extractedText = BinaryConverter.binaryToText(binaryMsg);
    
    if (isEncrypted) {
      if (!password) {
        throw new Error('This message is encrypted and requires a password.');
      }    

        try {
          return EncryptionService.decrypt(extractedText, password);
        } catch (decryptError) {
          throw new Error(`Decryption failed: ${decryptError.message}`);
        }
    }
  }

  static _embedData(pixels, pixelIndices, binaryData, width, protectedZones = []) {
    let bitIndex = 0;
    const channels = [0, 1, 2]; // R, G, B channels (fixed order)
    
    for (let i = 0; i < pixelIndices.length && bitIndex < binaryData.length; i++) {
      const pixelNum = pixelIndices[i];
      const x = pixelNum % width;
      const y = Math.floor(pixelNum / width);
      if (this._isInProtectedZone(x, y, protectedZones)) {
        continue; // Skip pixels in protected zones
      }
      const pixelPos = pixelIndices[i] * 4; // RGBA = 4 bytes per pixel
      
      // Embed up to 3 bits in each pixel (one per RGB channel)
      for (let c = 0; c < 3 && bitIndex < binaryData.length; c++) {
        const bit = parseInt(binaryData[bitIndex]);
        
        // Clear the LSB and set it to our bit
        pixels[pixelPos + channels[c]] = (pixels[pixelPos + channels[c]] & 0xFE) | bit;
        
        bitIndex++;
      }
    }
      if (bitIndex < binaryData.length) {
        throw new Error('Not enough space to embed message outside protected zones.');
      }
  }

  static _extractBitsAt(pixels, startBitIndex, numBits) {
    let bits = '';
    const channels = [0, 1, 2]; // R, G, B in fixed order
    
    // Calculate starting pixel and channel
    let pixelIndex = Math.floor(startBitIndex / 3);
    let channelOffset = startBitIndex % 3;
    
    for (let i = 0; i < numBits; i++) {
      const pixelPos = pixelIndex * 4; // RGBA = 4 bytes per pixel
      const channel = channels[channelOffset];
      
      // Extract the LSB from the appropriate channel
      bits += (pixels[pixelPos + channel] & 1).toString();
      
      // Move to next channel or pixel
      channelOffset = (channelOffset + 1) % 3;
      if (channelOffset === 0) {
        pixelIndex++;
      }
    }
    
    return bits;
  }
  static _isInProtectedZone(x, y, zones) {
  return zones.some(zone => (
    x >= zone.x && x < zone.x + zone.width &&
    y >= zone.y && y < zone.y + zone.height
  ));
}
}

module.exports = SteganoCore;