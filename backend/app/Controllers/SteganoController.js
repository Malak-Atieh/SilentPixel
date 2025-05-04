const SteganoService = require('../Services/SteganoService');
const { createResponse } = require('../Traits/response');

class SteganoController {
  static async encode(req, res) {
    try {
      const { message, password, generateQR, watermark } = req.body;
      const user = req.user;
      const imageFile = req.file; 
  
      if (!imageFile) {
        return createResponse(res, 400, 'Image file is required');
      }
      if (!message) {
        return createResponse(res, 400, 'Message is required');
      }
      if (!password) {
        return createResponse(res, 400, 'Password is required');
      }
      const finalWatermark = watermark
        ? `Watermarked for ${user.email} at ${new Date().toISOString()}`
        : null;

      const result = await SteganoService.encode({
        user.id,
        imageFile,
        message,
        password,
        finalWatermark,
        generateQR,
      });

      return createResponse(res, 200, 'Image encoded successfully', result);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }

  static async decode(req, res) {
    try {
      const { password } = req.body;
      const imageFile = req.file;
      const qrFile = req.files?.qr || null;
      const userId = req.user.id;

      const result = await SteganoService.decode(userId, imageFile, qrFile, password);
      return createResponse(res, 200, 'Message decoded successfully', result);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }
}

module.exports = SteganoController;
