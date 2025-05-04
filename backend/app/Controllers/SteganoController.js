const SteganoService = require('../Services/SteganoService');
const { createResponse } = require('../Traits/response');

class SteganoController {
  static async encode(req, res) {
    try {
      const { message, password, generateQR, watermark } = req.body;
      const user = req.user;
      const imageFile = req.file; 

      if (watermark){
        // default watermark: email + timestamp if user checks watermark
        const defaultWatermark = `Watermarked for ${user.email} at ${new Date().toISOString()}`;
      }
      
      const result = await SteganoService.encode({
        userId: user.id,
        imageFile,
        message,
        password,
        generateQR,
        watermark:  defaultWatermark ? defaultWatermark: watermark, 
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
      const userId = req.user.id;

      const result = await SteganoService.decode(userId, imageFile, password);
      return createResponse(res, 200, 'Message decoded successfully', result);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }
}

module.exports = SteganoController;
