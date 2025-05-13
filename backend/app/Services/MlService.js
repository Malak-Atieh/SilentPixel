const axios = require('axios');
const FormData = require('form-data');
const { AppError } = require('../Traits/errors');
class MLService {
  constructor() {
    this.apiUrl = process.env.ML_API_URL || 'http://localhost:5000/api';
    this.defaultTimeout = 30000;
  }

   async detectSteganography(imageBuffer) {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, { 
        filename: 'image.png',
        contentType: 'image/png' 
      });
      
      const response = await axios.post(
        `${this.apiUrl}/analyze`, 
        formData, 
        { 
          headers: { ...formData.getHeaders() },
          timeout: 30000 // 30 second timeout
        }
      );
      
      return response.data;
    } catch (error) {
      console.error('ML Service error:', error.message);
      if (error.response) {
        throw new AppError(`ML Service: ${error.response.data.message || 'Analysis failed'}`, 
                          error.response.status);
      }
      throw new AppError('ML Service unavailable. Please try again later.', 503);
    }
  }

    async detectBusyAreas(imageBuffer, sensitivity = 'medium') {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, { 
        filename: 'image.png', 
        contentType: 'image/png' 
      });
      formData.append('sensitivity', sensitivity);
      
      const response = await axios.post(
        `${this.apiUrl}/detect-busy-areas`, 
        formData, 
        { 
          headers: { ...formData.getHeaders() },
          timeout: 30000
        }
      );
      
      return response.data.busyAreas || [];
    } catch (error) {
      console.error('ML Service error:', error.message);
      return [];
    }
  }

}
module.exports = new MLService();
