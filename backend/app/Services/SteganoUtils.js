const SteganoCore = require('../utils/steganoFunctions/steganoCore');
const SteganoValidator = require('../Requests/SteganoValidator');
const {AppError} = require('../Traits/errors');
const crypto = require('crypto');

class SteganoUtils {

  static async embedMessage({ imageBuffer, message, password, busyAreas = [], protectedZones=[] }) {
    try {
      if (!imageBuffer || !Buffer.isBuffer(imageBuffer)) {
        throw new Error('Valid image buffer is required');
      }
      
      if (!message || typeof message !== 'string' || message.trim().length === 0) {
        throw new Error('Valid message string is required');
      }
      
      return await SteganoCore.embed(imageBuffer, message, password, busyAreas, protectedZones);
    } catch (error) {
      throw new AppError(`Embedding failed: ${error.message}`, 400);
    }
  }

  static async extractMessage(imageBuffer, password) {
    try {
      if (!imageBuffer || !Buffer.isBuffer(imageBuffer)) {
        throw new Error('Valid image buffer is required');
      }
      return await SteganoCore.extract(imageBuffer, password);
    } catch (error) {
      throw new AppError(`Extraction failed: ${error.message}`, 400);
    }
  }

  static async extractMessageWithQR(imageBuffer, password, qrData) {
    if (!qrData || !qrData.messageHash) {
      throw new Error('QR data must contain a messageHash for verification');
    }
    try {
      const message = await this.extractMessage(imageBuffer, password);

      const calculatedHash = this.generateMessageHash(message, password);
      if (calculatedHash !== qrData.messageHash) {
        throw new Error('Message integrity verification failed');
      }
      return message;
    } catch (error) {
      throw new AppError('Message integrity verification failed. The image may have been tampered with.', 400);
    }
  }

    static generateMessageHash(message, password) {
      if (!message) {
        throw new Error('Message is required for hash generation');
      }
      return crypto.createHash('sha256')
        .update(message + (password || ''))
        .digest('hex');
  }
}

module.exports = SteganoUtils;