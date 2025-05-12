const SteganoService = require('../Services/SteganoService');
const QRService = require('../Services/qrService');
const WatermarkService = require('../Services/WatermarkService');
const MLService = require('../Services/MLService');
const Image = require('../Models/Image');
const { createResponse } = require('../Traits/response');

class SteganoController {

  static async encode(req, res) {
    try {
      const processedImage = await SteganographyService.handleEncoding(req);
      
      // Return the processed image
      res.set('Content-Type', req.file.mimetype);
      return createResponse(res, 200, 'Image encoded successfully', processedImage);
    } catch (err) {
      return createResponse(res, err.status || 500, err.message);
    }
  }

    static async decode(req, res) {
    try {
      const message = await SteganographyService.handleDecoding(req);
      return createResponse(res, 200, 'Message decoded successfully', message);
    } catch (err) {
      return createResponse(res, err.status || 500, err.message);
    }
  }
/*
  static async encode(req, res) {
    
    try {
      const { message, password, addWatermark, addQRCode, busyAreas} = req.body;
      const user = req.user;

      if (!req.file) {
        return createResponse(res, 400, 'Image file is required');
      }

      if (!message) {
        return createResponse(res, 400, 'Message is required');
      }

      //image buffer
      const imageBuffer = req.file.buffer;  

      // Core steganography operation
      let processedImage = await SteganoService.embedMessage(
        imageBuffer,
        message,
        password,
        JSON.parse(busyAreas || '[]')
      );
      
      // Add watermark if requested
      if (addWatermark === 'true') {
        const watermarkData = {
          email: user.email,
          timestamp: new Date().toISOString()
        };
        processedImage = await WatermarkService.addWatermark(processedImage, watermarkData);
      }
      
      // Add QR code if requested
      if (addQRCode === 'true') {
        processedImage = await QRService.addQRCode(processedImage, { 
          messageHash: SteganoService.generateMessageHash(message, password),
          timestamp: new Date().toISOString()
        });
      }
      
      // Save metadata in database
      const imageDoc = new Image({
        userId: user._id,
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
          message = await SteganoService.extractMessageWithQR(req.file.buffer, password, qrData);
        }
      } catch (Error) {
        // QR extraction failed, proceed to standard extraction
        message = null;
      }
      
      // If QR extraction didn't work, try standard extraction
      if (!message) {
        message = await SteganoService.extractMessage(req.file.buffer, password);
      }
      
      return createResponse(res, 200, 'Message decoded successfully', message);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }

  async analyzeImage(req, res) {
    try {
      if (!req.file) {
        return createResponse(res, 400, 'Image is required');
      }
      
      const imageBuffer = req.file.buffer;

      // Use ML to detect if image likely contains hidden data
      const analysis = await MLService.detectSteganography(imageBuffer);
      
      // Extract watermark if present
      let watermarkData = null;
      try {
        watermarkData = await WatermarkService.extractWatermark(imageBuffer);
      } catch (error) {
        watermarkData = null; // No watermark found or error in extraction
      }
      
      // Final recommendation
      const response = {
        likelyContainsHiddenData: analysis.hiddenDataProbability > 0.7,
        hiddenDataProbability: analysis.hiddenDataProbability,
        watermarkData,
        recommendedBusyAreas: analysis.busyAreas || []
      };

      return createResponse(res, 200, 'Image analysis complete', response);
    } catch (error) {
      return errorHandler(error, res);
    }
  }*/
}

module.exports = SteganoController;
