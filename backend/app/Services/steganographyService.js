const SteganoService = require('./SteganoService');
const QRService = require('./qrService');
const WatermarkService = require('./WatermarkService');
const MLService = require('./MLService');
const Image = require('../Models/Image');

class SteganographyService {
  static async handleEncoding(req) {
    const { message, password, addWatermark, addQRCode, busyAreas } = req.body;
    const user = req.user;

    if (!req.file) throw new Error('Image file is required');
    if (!message) throw new Error('Message is required');

    const imageBuffer = req.file.buffer;
    let processedImage = await SteganoService.embedMessage(
      imageBuffer,
      message,
      password,
      JSON.parse(busyAreas || '[]')
    );

    if (addWatermark === 'true') {
      processedImage = await WatermarkService.addWatermark(processedImage, {
        email: user.email,
        timestamp: new Date().toISOString(),
      });
    }

    if (addQRCode === 'true') {
      processedImage = await QRService.addQRCode(processedImage, {
        messageHash: SteganoService.generateMessageHash(message, password),
        timestamp: new Date().toISOString(),
      });
    }

    const imageDoc = new Image({
      userId: user._id,
      filename: req.file.originalname,
      contentType: req.file.mimetype,
      hasWatermark: addWatermark === 'true',
      hasQRCode: addQRCode === 'true',
      processingDate: new Date(),
      messageLength: message.length,
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
        return await SteganoService.extractMessageWithQR(buffer, password, qrData);
      }
    } catch (_) {
      // QR decode failed; fallback
    }

    return await SteganoService.extractMessage(buffer, password);
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
