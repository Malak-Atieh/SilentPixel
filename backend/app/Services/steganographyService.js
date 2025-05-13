const SteganoUtils = require('./SteganoUtils');
const QRService = require('./qrService');
const WatermarkService = require('./WatermarkService');
const MLService = require('./MLService');
const Image = require('../Models/Image');
const { ValidationError, AppError } = require('../Traits/errors');

class SteganographyService {

  static async handleEncoding(req) {
    const { message, password, addWatermark, addQRCode, busyAreas } = req.body;
    const user = req.user;

    if (!req.file){
      throw new ValidationError('Image file is required');
    } 
    if (!message){
      throw new ValidationError('Message is required');
    }

    const imageBuffer = req.file.buffer;
    let processedImage ;
    try {
      processedImage = await SteganoUtils.embedMessage({
        imageBuffer,
        message,
        password,
        busyAreas: JSON.parse(busyAreas || '[]'),
      });

      if (addWatermark === 'true') {
        processedImage = await WatermarkService.addWatermark(processedImage, {
          email: user.email,
          timestamp: new Date().toISOString(),
        });
      }

    } catch (error){
      throw new AppError(`Image analysis failed: ${error.message}`, 
        error.status || 500);
    }

  }
}

module.exports = SteganographyService;
