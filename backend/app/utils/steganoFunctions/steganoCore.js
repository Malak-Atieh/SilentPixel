require('dotenv').config();
const BinaryConverter = require('./binaryConverter');
const PixelSelector = require('./pixelSelector');
const EncryptionService = require('../../Services/EncryptionService');
const ImageProcessor = require('../imageProcessor');

class SteganoCore {


  static async embed(imageBuffer, message, password, busyAreas = [], protectedZones = [],options = {}) {

    const { ttl } = options;
    const timestamp = Math.floor(Date.now() / 1000);
    const expiryTime = ttl ? timestamp + ttl : 0;
    const { image } = await ImageProcessor.loadImage(imageBuffer);
      
    const imageData = await ImageProcessor.getImageData(image);
      
    const encryptedMsg = EncryptionService.encrypt(message, password);
    const binaryMsg = BinaryConverter.textToBinary(encryptedMsg);

    // Create header: signature(8) + length(32) + is_encrypted(1) + has_ttl(1) + ttl(32)
    const header = process.env.SIGNATURE + 
                  BinaryConverter.numberToBinary(binaryMsg.length, 32) + 
                  (password ? "1" : "0") +
                 (ttl ? "1" : "0") +
                 (ttl ? BinaryConverter.numberToBinary(expiryTime, 32) : "");

    const dataToHide = header + binaryMsg;

    const pixelsNeeded = Math.ceil(dataToHide.length / 3);
    const totalPixels = imageData.width * imageData.height;
    
    const pixelIndices = Array.from({length: totalPixels}, (_, i) => i);
      if (pixelsNeeded > totalPixels) {
        throw new Error(`Insufficient capacity: need ${pixelsNeeded} pixels, but only ${totalPixels} available.`);
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

  static async embedMultiple(imageBuffer, messages, passwords = [], busyAreas = [], protectedZones = [], options = {}) {
    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      throw new Error('At least one message is required');
    }

    if (passwords && passwords.length > 0 && passwords.length !== messages.length) {
      throw new Error('Number of passwords must match number of messages');
    }
    
    const { image } = await ImageProcessor.loadImage(imageBuffer);
    const imageData = await ImageProcessor.getImageData(image);
    
    // Prepare all messages
    const encryptedMessages = messages.map((msg, i) => {
      const pwd = passwords && passwords.length > 0 ? passwords[i] : null;
      return {
        encrypted: pwd ? EncryptionService.encrypt(msg, pwd) : msg,
        isEncrypted: !!pwd
      };
    });
    
    // Create a composite message with headers for each
    let compositeBinary = "";
    let headers = "";
    

    
    encryptedMessages.forEach(msg => {
      const binaryMsg = BinaryConverter.textToBinary(msg.encrypted);
      headers += BinaryConverter.numberToBinary(binaryMsg.length, 32) +
                (msg.isEncrypted ? "1" : "0");
      compositeBinary += binaryMsg;
    });

    const header = process.env.SIGNATURE + 
                  BinaryConverter.numberToBinary(messages.length, 8) +
                  headers;
    
    const dataToHide = header + compositeBinary;
    
    const pixelsNeeded = Math.ceil(dataToHide.length / 3);
    const totalPixels = imageData.width * imageData.height;
    
    const pixelIndices = Array.from({length: totalPixels}, (_, i) => i);
    if (pixelsNeeded > totalPixels) {
      throw new Error(`Insufficient capacity: need ${pixelsNeeded} pixels, but only ${totalPixels} available.`);
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

    if (signature !== process.env.SIGNATURE) {
      throw new Error('Invalid signature. This image does not appear to contain hidden data.');
    }
    
    const binaryLength = this._extractBitsAt(imageData.data, 8, 32);
    const messageLength = parseInt(binaryLength, 2);

    
    const isEncrypted = this._extractBitsAt(imageData.data, 40, 1) === "1";
    const hasTTL = this._extractBitsAt(imageData.data, 41, 1) === "1";
    
    let ttlOffset = 0;
    let expiryTime = 0;
    
    if (hasTTL) {
      expiryTime = parseInt(this._extractBitsAt(imageData.data, 42, 32), 2);
      ttlOffset = 32;
      
      const currentTime = Math.floor(Date.now() / 1000);
      if (currentTime > expiryTime) {
        throw new Error('This message has expired and can no longer be viewed');
      }
    }

    const binaryMsg = this._extractBitsAt(imageData.data, 42 + ttlOffset, messageLength);
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
    return extractedText;
  }

  static async extractMultiple(imageBuffer, password) {
    const { image } = await ImageProcessor.loadImage(imageBuffer);
    const imageData = await ImageProcessor.getImageData(image);
    imageBuffer = null;

    const signature = this._extractBitsAt(imageData.data, 0, 8);
    if (signature !== process.env.SIGNATURE) {
      throw new Error('Invalid signature');
    }
    
    const messageCount = parseInt(this._extractBitsAt(imageData.data, 8, 8), 2);
    
    let bitPosition = 16;
    const messageInfos = [];
    
    for (let i = 0; i < messageCount; i++) {
      const length = parseInt(this._extractBitsAt(imageData.data, bitPosition, 32), 2);
      bitPosition += 32;
      
      const isEncrypted = this._extractBitsAt(imageData.data, bitPosition, 1) === "1";
      bitPosition += 1;
      
      messageInfos.push({
        length,
        isEncrypted,
        startBit: bitPosition
      });
      
      bitPosition += length;
    }
    
    const results = [];
    
    for (const info of messageInfos) {
      try {
        const binaryMsg = this._extractBitsAt(imageData.data, info.startBit, info.length);
        const encryptedMsg = BinaryConverter.binaryToText(binaryMsg);
        
        if (info.isEncrypted) {
          if (!password) {
            results.push(null);
            continue;
          }
          
          try {
            const decrypted = EncryptionService.decrypt(encryptedMsg, password);
            results.push(decrypted);
          } catch (e) {
            console.log("Failed to decrypt a message:", e.message);
            results.push(null);
          }
        } else {
          results.push(encryptedMsg);
        }
      } catch (e) {
        console.log("Failed to extract a message:", e.message);
        results.push(null);
      }
    }
    image.options = null;
    return results;
  }


  static _embedData(pixels, pixelIndices, binaryData, width, protectedZones = []) {
    let bitIndex = 0;
    const channels = [0, 1, 2];
    
    for (let i = 0; i < pixelIndices.length && bitIndex < binaryData.length; i++) {
      const pixelNum = pixelIndices[i];
      const x = pixelNum % width;
      const y = Math.floor(pixelNum / width);
      if (this._isInProtectedZone(x, y, protectedZones)) {
        continue;
      }
      const pixelPos = pixelIndices[i] * 4; 
      
      for (let c = 0; c < 3 && bitIndex < binaryData.length; c++) {
        const bit = parseInt(binaryData[bitIndex]);
        
        pixels[pixelPos + channels[c]] = (pixels[pixelPos + channels[c]] & 0xFE) | bit;
        
        bitIndex++;
      }
    }
    if (bitIndex < binaryData.length) {

      const lowPriorityZones = protectedZones.filter(zone => zone.priority !== 'high');
      const highPriorityZones = protectedZones.filter(zone => zone.priority === 'high');
      
      for (let i = 0; i < pixelIndices.length && bitIndex < binaryData.length; i++) {
        const pixelNum = pixelIndices[i];
        const x = pixelNum % width;
        const y = Math.floor(pixelNum / width);
        
        if (this._isInSpecificProtectedZones(x, y, highPriorityZones)) {
          continue;
        }
        
        const pixelPos = pixelIndices[i] * 4;
        
        for (let c = 0; c < 3 && bitIndex < binaryData.length; c++) {
          const bit = parseInt(binaryData[bitIndex]);
          pixels[pixelPos + channels[c]] = (pixels[pixelPos + channels[c]] & 0xFE) | bit;
          bitIndex++;
        }
      }
    }
      if (bitIndex < binaryData.length) {
        throw new Error('Not enough space to embed message outside protected zones.');
      }
  }

  static _extractBitsAt(pixels, startBitIndex, numBits) {
    let bits = '';
    const channels = [0, 1, 2]; 
    
    let pixelIndex = Math.floor(startBitIndex / 3);
    let channelOffset = startBitIndex % 3;
    
    for (let i = 0; i < numBits; i++) {
      const pixelPos = pixelIndex * 4; 
      const channel = channels[channelOffset];
      
      bits += (pixels[pixelPos + channel] & 1).toString();
      
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

static _isInSpecificProtectedZones(x, y, zones) {
  return zones.some(zone => (
    x >= zone.x && x < zone.x + zone.width &&
    y >= zone.y && y < zone.y + zone.height
  ));
}
}


module.exports = SteganoCore;