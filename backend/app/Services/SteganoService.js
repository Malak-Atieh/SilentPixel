const StegoImage = require('../Models/StegoImage');
const { callPythonML } = require('./MLService');

class SteganoService {
  static async encode(userId, imageFile, message, password, watermark, generateQR) {
    // Call Python ML microservice
    const { encodedPath, watermarkId, qrPath  } = await callPythonML({
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
      originalPath: imageFile.originalname,
      encodedPath,
      watermarkId,
      qrPath: qrPath || null,
    });

    return { encodedPath, watermarkId, qrPath  };
  }

  static async decode(userId, imageFile, password = null) {
    // Call Python ML microservice for decoding
    const { message, watermarkId } = await callPythonML({
        image: imageFile.buffer,
        password
    });

    //  log the decoding
    await StegoImage.create({
        userId,
        originalPath: imageFile.originalname,
        encodedPath: null,
        watermarkId,
        decoded: true
    });

    return { message, watermarkId };  
    }
}

module.exports = SteganoService;