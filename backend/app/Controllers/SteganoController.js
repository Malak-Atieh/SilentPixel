const SteganographyService = require('../Services/SteganographyService');
const { createResponse } = require('../Traits/response');

class SteganoController {

  static async encode(req, res) {
    try {

      const processedImage = await SteganographyService.handleEncoding(req);
      
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
