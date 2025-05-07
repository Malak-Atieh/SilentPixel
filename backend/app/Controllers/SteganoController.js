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
      return createResponse(res, 200, 'Image encoded successfully', processedImage);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }

  static async decode(req, res) {
    try {
      const { password } = req.body;
      
      if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
      }
      
      // First try QR extraction if present
      let message;
      try {
        const qrData = await QRService.extractQRData(req.file.buffer);
        if (qrData) {
          // Verify password against message hash in QR
          message = await StegoService.extractMessageWithQR(req.file.buffer, password, qrData);
        }
      } catch (qrError) {
        // QR extraction failed or not present, continue with standard extraction
      }
      
      // If QR extraction didn't work, try standard extraction
      if (!message) {
        message = await StegoService.extractMessage(req.file.buffer, password);
      }
      
      return createResponse(res, 200, 'Message decoded successfully', message);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }

  async analyzeImage(req, res) {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No image uploaded' });
      }
      
      // Use ML to detect if image likely contains hidden data
      const analysis = await MLService.detectSteganography(req.file.buffer);
      
      // Extract watermark if present
      let watermarkData = null;
      try {
        watermarkData = await WatermarkService.extractWatermark(req.file.buffer);
      } catch (watermarkError) {
        // No watermark or extraction failed
      }
      
      return res.json({
        likelyContainsHiddenData: analysis.hiddenDataProbability > 0.7,
        hiddenDataProbability: analysis.hiddenDataProbability,
        watermarkData: watermarkData,
        recommendedAreas: analysis.busyAreas.slice(0, 3) // Top 3 areas for hiding
      });
    } catch (error) {
      return errorHandler(error, res);
    }
  }
}

module.exports = SteganoController;
