const SteganoCore = require('../utils/steganoFunctions/steganoCore');
const SteganoValidator = require('../Requests/SteganoValidator');
const {AppError} = requie('../Traits/errors');

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
      SteganoValidator.verifyMessageHash(message, password, qrData.messageHash);
      return message;
    } catch (error) {
      throw new AppError('Message integrity verification failed. The image may have been tampered with.', 400);
    }
  }
}

module.exports = SteganoUtils;