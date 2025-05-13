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
    let processedImage = await SteganoUtils.embedMessage({
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

    if (addQRCode === 'true') {
      processedImage = await QRService.addQRCode(processedImage, {
        messageHash: SteganoUtils.generateMessageHash(message, password),
        timestamp: new Date().toISOString(),
      });
    }

    const imageDoc = new Image({
        userId: user._id,
        originalImage: {
          filename: req.file.originalname,
          contentType: req.file.mimetype,
          size: req.file.size
        },
        stegoDetails: {
          hasHiddenContent: true,
          messageLength: message.length,
          isPasswordProtected: !!password
        },
        watermark: {
          hasWatermark: addWatermark === 'true',
          watermarkType: addWatermark === 'true' ? 'invisible' : 'none',
          timestamp: new Date()
        },
        qrCode: {
          hasQRCode: addQRCode === 'true'
        },
        processingDetails: {
          processedAt: new Date(),
          stegoMethod: 'lsb'
        }
    });

    await imageDoc.save();

    return processedImage;
  }

  static async handleDecoding(req) {
    const { password } = req.body;
    if (!req.file) throw new Error('No image uploaded');

    const buffer = req.file.buffer;

    try {
      const qrData = await QRService.extractQRData(buffer);
      if (qrData) {
        return await SteganoUtils.extractMessageWithQR(buffer, password, qrData);
      }
    } catch (_) {
      // QR decode failed; fallback
    }

    return await SteganoUtils.extractMessage(buffer, password);
  }

  static async handleAnalysis(req) {
    if (!req.file) throw new Error('Image is required');
    const imageBuffer = req.file.buffer;

    const analysis = await MLService.detectSteganography(imageBuffer);

    let watermarkData = null;
    try {
      watermarkData = await WatermarkService.extractWatermark(imageBuffer);
    } catch (_) {
      watermarkData = null;
    }

    return {
      likelyContainsHiddenData: analysis.hiddenDataProbability > 0.7,
      hiddenDataProbability: analysis.hiddenDataProbability,
      watermarkData,
      recommendedBusyAreas: analysis.busyAreas || [],
    };
  }
}

module.exports = SteganographyService;
