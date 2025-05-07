const StegoImage = require('../Models/Image');
const MLService= require('./MLService');
const fs = require('fs');
const path = require('path');
const {LSBEncoder, LSBDecoder} = require('../utils/steganography');
class steganoService {

  constructor() {
    this.MLService = new MLService();
  }

  async analyzeImage(imagePath, userId) {
    try{
      //call the ML service to analyze the image and detct the busy areas
      const analysisResult = await this.MLService.analyzeBusyAreas(imagePath);
      
      // Save the analysis result to the database
      const stegoImage = new StegoImage({
        userId,
        originalImagePath: imagePath,
        busyAreasMap: analysisResult.busyAreasMap,
      });

      await stegoImage.save();
      return stegoImage;
    }catch(error){
      throw new Error(`Image analysis failed: ${error.message}`)
    }


  }

  static async encodeMessage(imageId, message, password, options={}) {
    try{
      const {
        addWatermark = false, 
        addQrCode= false
      } = options;
      
      //get saved image from the database
      const stegoImage = await StegoImage.findById(imageId);
      if (!stegoImage) throw new Error('Image not found');

      //prepare the image for encoding
      const encodingData = {
        message,
        password,
        busyAreasMap: stegoImage.busyAreasMap,
      };

      //add watermark if asked
      if (addWatermark){
        const timestamp = new Date().toISOString();
        encodingData.watermark = `${userEmail}/${timestamp}`;
      }
      if (addQrCode){
        encodingData.qrCodeData = message;
      }

      //do encoding
      const encoder = new LSBEncoder();
      const encodedImagePath=path.join(
        path.dirname(stegoImage.originalImagePath),
        `encoded_${path.basename(stegoImage.originalImagePath)}`
      );

      await encoder.encode(
        stegoImage.originalImagePath,
        encodedImagePath,   
        encodingData
      );

      //update db record
      stegoImage.encodedImagePath = encodedImagePath;
      stegoImage.hasWatermark = addWatermark;
      stegoImage.hasQrCode = addQrCode;
      stegoImage.metaData={
        encodingMethod: 'LSB',
        messageLength: message.length,
        watermarkInfo: addWatermark ? `${userEmail}|timestamp` : null,
        qrCodeData: addQrCode ? 'QR data embeded' : null,
      }

      await stegoImage.save();
      return stegoImage;
    } catch (error) {
      throw new Error(`Encoding failed: ${error.message}`);
    }   

  }

  static async decodeMessage(imageFile, password ) {
    try{
      const decode = new LSBDecoder();
      const result = await decode.decode(imageFile, password);

      return {
        message: result.message,
        watermark: result.watermark || null,
        qrCodeData: result.qrCodeData || null,
      }
    } catch (error) {
      throw new Error(`Decoding failed: ${error.message}`);
    }
  }

  async detectSteganography(imagePath) {
    try{
      const detectionResult = this.MLService.detectHiddenData(imagePath);
      return detectionResult;
    }catch (error) {
      throw new Error(`Steganography detection failed: ${error.message}`);
    }
  }
}

module.exports = steganoService;