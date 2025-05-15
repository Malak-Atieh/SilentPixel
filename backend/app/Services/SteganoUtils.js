const SteganoCore = require('../utils/steganoFunctions/steganoCore');
const SteganoValidator = require('../Requests/SteganoValidator');
const {AppError} = require('../Traits/errors');

class SteganoUtils {

  static async embedMessage({ imageBuffer, message, password, busyAreas = [] }) {
    try {
      return await SteganoCore.embed(imageBuffer, message, password, busyAreas);
    } catch (error) {
      throw new AppError(`Embedding failed: ${error.message}`, 400);
    }
  }

  static async extractMessage(imageBuffer, password) {
    try {
      return await SteganoCore.extract(imageBuffer, password);
    } catch (error) {
      throw new AppError(`Extraction failed: ${error.message}`, 400);
    }
  }

  static async extractMessageWithQR(imageBuffer, password, qrData) {
    const message = await this.extractMessage(imageBuffer, password);
    try {
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
      const crypto = require('crypto');
      return crypto.createHash('sha256')
        .update(message + (password || ''))
        .digest('hex');
  }
}

module.exports = SteganoUtils;