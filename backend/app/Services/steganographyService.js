const SteganoUtils = require('./SteganoUtils');
const QRService = require('./qrService');
const WatermarkService = require('./WatermarkService');
const MLService = require('./MLService');
const StegoImage = require('../Models/Image');
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
    let processedImage = imageBuffer ;
    const protectedZones = [];
    try {
      
      if (addWatermark === 'true') {
        wmResult  = await WatermarkService.addWatermark(processedImage, {
          email: user.email,
          timestamp: new Date().toISOString(),
        });
        processedImage = wmResult.data;
        if (wmResult.region) protectedZones.push(wmResult.region);
      }

      if (addQRCode === 'true') {
        const qrResult  = await QRService.addQRCode(processedImage, {
          messageHash: SteganoUtils.generateMessageHash(message, password),
          timestamp: new Date().toISOString(),
        });
        processedImage = qrResult.data;
        if (qrResult.region) protectedZones.push(qrResult.region);
      }

      processedImage = await SteganoUtils.embedMessage({
        imageBuffer: processedImage, 
        message,
        password,
        busyAreas: JSON.parse(busyAreas || '[]'),
        protectedZones
      });


      const imageDoc = new StegoImage({
          userId: user.userId,
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
    } catch (error) {
      throw new AppError(`Encoding failed: ${error.message}`, 
        error.status || 500);
    }
  }

  static async handleDecoding(req) {
    const { password } = req.body;

    if (!req.file)  {
      throw new ValidationError('No image uploaded');
    }

    const buffer = req.file.buffer;
    const result = {
      message: null,
      watermark: null,
      qrData: null
    };
    try {
      try{
        const qrData = await QRService.extractQRData(buffer);
        if (qrData) {
          result.qrData = qrData;
          result.message = await StegoUtils.extractMessageWithQR(buffer, password, qrData);
        }
      } catch (qrError) {
        // QR extraction failed, falling back to standard extraction
      }
      if (!result.message) {
        result.message = await StegoUtils.extractMessage(buffer, password);
      }
      try{
         result.watermark = await WatermarkService.extractWatermark(buffer);
      }catch(watermarkError){
        console.log("Watermark extraction failed:", watermarkError.message);
      }
    return result;
    } catch(error){
      throw new AppError(`Decoding failed: ${error.message}`, 
      error.status || 400);
    }
  }

  static async handleAnalysis(req) {
    if (!req.file) {
      throw new ValidationError('Image is required');
    }

    const imageBuffer = req.file.buffer;
    const authToken = req.header('Authorization');

    try {
      const analysis = await MLService.detectSteganography(imageBuffer, authToken);
      
      let watermarkData = null;
      let qrData = null;
      
      // Try to extract watermark
      try {
        watermarkData = await WatermarkService.extractWatermark(imageBuffer);
      } catch (watermarkError) {
        // Continue without watermark data
      }
      
      // Try to extract QR code
      try {
        qrData = await QRService.extractQRCode(imageBuffer);
      } catch (qrError) {
        // Continue without QR data
      }

      return {
        likelyContainsHiddenData: analysis.hiddenDataProbability > 0.7,
        hiddenDataProbability: analysis.hiddenDataProbability,
        watermarkData,
        qrData,
        recommendedBusyAreas: analysis.busyAreas || [],
      };
    } catch (error) {
      throw new AppError(`Image analysis failed: ${error.message}`, 
        error.status || 500);
    }
  }

}

module.exports = SteganographyService;
