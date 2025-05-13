const SteganoCore = require('../utils/steganoFunctions/steganoCore');
const SteganoValidate = require('../Requests/SteganoValidate');

class SteganoService {
  static async embedMessage(imageBuffer, message, password, busyAreas = []) {
    try {
      return await SteganoCore.embed(imageBuffer, message, password, busyAreas);
    } catch (error) {
      throw new Error(`Encoding failed: ${error.message}`);
    }
  }

  static async extractMessage(imageBuffer, password) {
    try {
      return await SteganoCore.extract(imageBuffer, password);
    } catch (error) {
      throw new Error(`Decoding failed: ${error.message}`);
    }
  }

  static async extractMessageWithQR(imageBuffer, password, qrData) {
    const message = await this.extractMessage(imageBuffer, password);
    SteganoValidate.verifyMessageHash(message, password, qrData.messageHash);
    return message;
  }
}

module.exports = SteganoService;