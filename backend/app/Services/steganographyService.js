const SteganoUtils = require('./SteganoUtils');
const QRService = require('./qrService');
const WatermarkService = require('./WatermarkService');
const MLService = require('./MLService');
const Image = require('../Models/Image');
const { ValidationError, AppError } = require('../Traits/errors');

class SteganographyService {

  static async handleEncoding(req) {
  }
}

module.exports = SteganographyService;
