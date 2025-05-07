const SteganoService = require('../Services/SteganoService');
const QRService = require('../Services/qrService');
const WatermarkService = require('../Services/WatermarkService');
const MLService = require('../Services/MLService');
const Image = require('../Models/Image');
const { createResponse } = require('../Traits/response');

class SteganoController {

  static async encode(req, res) {
    
    try {
      const { message, password, addWatermark, addQRCode} = req.body;
      const user = req.user;

      if (!req.file) {
        return createResponse(res, 400, 'Image file is required');
      }

      if (!message) {
        return createResponse(res, 400, 'Message is required');
      }

      if (!password) {  
        return createResponse(res, 400, 'Password is required');
      }

      //image buffer
      const imageBuffer = req.file.buffer;  

      // Use ML service to detect busy areas for optimal hiding
      const busyAreas = await MLService.detectBusyAreas(imageBuffer);
      
      // Core steganography operation
      let processedImage = await StegoService.embedMessage(
        imageBuffer,
        message,
        password,
        busyAreas
      );
      
      // Add watermark if requested
      if (addWatermark === 'true') {
        const watermarkData = {
          email: req.user.email,
          timestamp: new Date().toISOString()
        };
        processedImage = await WatermarkService.addWatermark(processedImage, watermarkData);
      }
      
      // Add QR code if requested
      if (addQRCode === 'true') {
        processedImage = await QRService.addQRCode(processedImage, { 
          messageHash: StegoService.generateMessageHash(message, password),
          timestamp: new Date().toISOString()
        });
      }
      
      // Save metadata in database
      const imageDoc = new Image({
        userId: req.user._id,
        filename: req.file.originalname,
        contentType: req.file.mimetype,
        hasWatermark: addWatermark === 'true',
        hasQRCode: addQRCode === 'true',
        processingDate: new Date(),
        messageLength: message.length
      });
      
      await imageDoc.save();
      
      // Return the processed image
      res.set('Content-Type', req.file.mimetype);
      return res.send(processedImage);
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
