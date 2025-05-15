const SteganographyService = require('../Services/steganographyService');
const { createResponse } = require('../Traits/response');
const fs = require('fs');
const path = require('path');
class SteganoController {

  static async encode(req, res) {

    try {
const user = req.user;
      const processedImage = await SteganographyService.handleEncoding(req);
      const base64Image = `data:${req.file.mimetype};base64,${processedImage.toString('base64')}`;
      const fileName = `${user.userId}_${Date.now()}.png`;
      const filePath = path.join(__dirname, '../../storage/uploads', fileName);
      fs.writeFileSync(filePath, processedImage);

      const downloadUrl = `${process.env.BASE_URL}/download/${fileName}`;
      res.set('Content-Type', req.file.mimetype);

      return res.status(200).send(processedImage);

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

  static async analyzeImage(req, res) {

    try {

      const analysis = await SteganographyService.handleAnalysis(req);
      
      return createResponse(res, 200, 'Image analysis complete', analysis);
    
    } catch (err) {

      return createResponse(res, err.status || 500, err.message);
    
    }

  }

}

module.exports = SteganoController;
