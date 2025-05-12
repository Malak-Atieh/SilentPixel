const axios = require('axios');
const FormData = require('form-data');
const { AppError } = require('../Traits/errors');
class MLService {
  constructor() {
    this.apiUrl = process.env.ML_API_URL || 'http://localhost:5000/api';
  }

   async detectSteganography(imageBuffer) {

  }

}
module.exports = {  };
