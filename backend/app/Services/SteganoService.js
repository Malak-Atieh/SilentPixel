const StegoImage = require('../Models/StegoImage');
const MLService= require('./MLService');
const fs = require('fs');
const path = require('path');
const {LSBEncode, LSBDecoder} = require('../utils/steganography');
class SteganoService {
  static async encode(userId, imageFile, message, password, watermark, generateQR) {
    // Call Python ML microservice
    const { encodedUrl, watermark, qrPath  } = await callPythonML({
      image: imageFile.buffer,
      message,
      password,
      watermark,
      generateQR,
      type: 'encode',
    });

    // Save it to my database
    const stegoImage = await StegoImage.create({
      userId,
      type: 'encode',
      originalUrl: imageFile.originalname,
      encodedUrl,
      watermark,
      qrPath: qrPath || null,
      message
    });

    return { encodedUrl, watermark, qrPath  };
  }

  static async decode(userId, imageFile, qrFile=null, password ) {
    // Call Python ML microservice for decoding
    const { message, watermark } = await callPythonML({
        image: imageFile.buffer,
        qrImage: qrFile?.buffer,
        type: 'decode',
        password
    });

    //  log the decoding
    await StegoImage.create({
        userId,
        type: 'decode',
        originalPath: imageFile.originalname,
        encodedPath: null,
        watermark,
        qrPath: qrFile?.originalname || null,
        message
    });

    return { message, watermark };  
    }
}

module.exports = SteganoService;