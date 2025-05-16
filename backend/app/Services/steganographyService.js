const SteganoUtils = require('./SteganoUtils');
const QRService = require('./qrService');
const WatermarkService = require('./WatermarkService');
const MLService = require('./MLService');
const StegoImage = require('../Models/Image');
const { ValidationError, AppError } = require('../Traits/errors');

class SteganographyService {

  static async handleEncoding(req) {
    const { message , password, addWatermark, addQRCode, busyAreas } = req.body;
    const user = req.user;

    if (!req.file){
      throw new ValidationError('Image file is required');
    } 

    const imageBuffer = req.file.buffer;
    let processedImage = imageBuffer ;
    const protectedZones = [];
    try {
      
      if (addWatermark === 'true') {
        const wmResult  = await WatermarkService.addWatermark(processedImage, {
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
    const options = {
      ttl: req.body.ttl ? parseInt(req.body.ttl) : null,
    };

    if (req.body.messages) {
      try{
        const messages = JSON.parse(req.body.messages);
        const passwords = req.body.passwords ? JSON.parse(req.body.passwords) : null;

        if (!Array.isArray(messages)) {
          throw new ValidationError('Messages must be an array');
        }
            
        if (passwords && passwords.length !== messages.length) {
          throw new ValidationError('Number of passwords must match number of messages');
        }

        processedImage = await SteganoUtils.embedMultipleMessages({
          imageBuffer: processedImage, 
          messages,
          passwords,
          busyAreas: JSON.parse(busyAreas || '[]'),
          protectedZones,
          options
        });        
      } catch (parseError) {
          throw new ValidationError(`Error parsing messages or passwords: ${parseError.message}`);
      }
    } else {
      processedImage = await SteganoUtils.embedMessage({
        imageBuffer: processedImage, 
        message,
        password,
        busyAreas: JSON.parse(busyAreas || '[]'),
        protectedZones,
        options
      });
    }

      const imageDoc = new StegoImage({
          userId: user.userId,
          originalImage: {
            filename: req.file.originalname,
            contentType: req.file.mimetype,
            size: req.file.size
          },
          stegoDetails: {
            hasHiddenContent: true,
            messageLength: message ? message.length : (req.body.messages ? JSON.parse(req.body.messages).reduce((sum, m) => sum + m.length, 0) : 0),
            isPasswordProtected: !!password || (req.body.passwords && JSON.parse(req.body.passwords).some(p => !!p))
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
      messages: null,
      watermark: null,
      qrData: null
    };
    
    try {
      
      //try qr extracting if it exits in image
      try{
        const qrData = await QRService.extractQRCode(buffer);
        if (qrData) {
          result.qrData = qrData;
          result.message = await SteganoUtils.extractMessageWithQR(buffer, password, qrData);
        }
      } catch (qrError) {
        console.log("QR extraction failed:", qrError.message);
      }

      //try watermark extracting if it exits in image
      try{
         result.watermark = await WatermarkService.extractWatermark(buffer);
      } catch (watermarkError){
        console.log("Watermark extraction failed:", watermarkError.message);
      }

      //try extracting multiple messages if it applies
      try {
        const messages = await SteganoUtils.extractMultipleMessages(buffer, password);
        if (messages && messages.length > 0 && messages.some(msg => msg !== null)) {
          result.messages = messages;
          // If we have only one message, put it in the message field too for backwards compatibility
          if (messages.length === 1) {
            result.message = messages[0];
          }
          return result;
        }
      } catch (multipleError) {
        console.log("Multiple message extraction failed:", multipleError.message);
      }

            try {
        const message = await SteganoUtils.extractMessage(buffer, password);
        if (message) {
          result.message = message;
        }
      } catch (singleError) {
        console.log("Single message extraction failed:", singleError.message);
        throw new AppError("Failed to extract any messages from the image. " + 
                           (password ? "Please check your password." : "The image may not contain hidden data."), 400);
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
      

      try {
        watermarkData = await WatermarkService.extractWatermark(imageBuffer);
      } catch (watermarkError) {
        console.log("no watermark included:", watermarkError);
      }
      

      try {
        qrData = await QRService.extractQRCode(imageBuffer);
      } catch (qrError) {
        console.log("no qr included:", qrError);
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
