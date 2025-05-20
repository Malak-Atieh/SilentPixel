const axios = require('axios');
const FormData = require('form-data');
const { AppError } = require('../Traits/errors');
class MLService {
  constructor() {
    this.apiUrl = process.env.ML_API_URL || 'http://localhost:5000/api';
    this.defaultTimeout = 30000;
  }

   async detectSteganography(imageBuffer, authToken) {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, { 
        filename: 'image.png',
        contentType: 'image/png' 
      });
      
      const headers = {
      ...formData.getHeaders(),
      'Authorization': authToken || process.env.ML_API_KEY 
      };

      const response = await axios.post(
        `${this.apiUrl}/analyze`, 
        formData, 
        { 
          headers: headers,
          timeout: this.defaultTimeout
        }
      );
      
      return response.data;
    } catch (error) {
      console.error('ML Service error:', error.message);
      if (error.response) {
        throw new AppError(
          `ML Service: ${error.response.data.message || 'Analysis failed'}`, 
                          error.response.status
                        );
      }
      throw new AppError('ML Service unavailable. Please try again later.', 503);
    }
  }

    async detectBusyAreas(imageBuffer, authToken, sensitivity = 'medium') {
    try {
      const formData = new FormData();
      formData.append('image', imageBuffer, { 
        filename: 'image.png', 
        contentType: 'image/png' 
      });
      formData.append('sensitivity', sensitivity);
      
      const headers = {
      ...formData.getHeaders(),
      'Authorization': authToken || process.env.ML_API_KEY // Use passed token or fallback to API key
      };

      const response = await axios.post(
        `${this.apiUrl}/detect-busy-areas`, 
        formData, 
        { 
          headers: headers,
          timeout: this.defaultTimeout
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
