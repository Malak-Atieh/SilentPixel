const SteganoService = require('../Services/SteganoService');
const { createResponse } = require('../Traits/response');

class SteganoController {
  static async encode(req, res) {
    try {
      const { message, password } = req.body;
      const imageFile = req.file; 
      const userId = req.user.id; 

      const result = await SteganoService.encode(userId, imageFile, message, password);
      return createResponse(res, 200, 'Image encoded successfully', result);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }

  static async decode(req, res) {
    try {
      const { password } = req.body;
      const imageFile = req.file;
      const userId = req.user.id;

      const result = await SteganoService.decode(userId, imageFile, password);
      return createResponse(res, 200, 'Message decoded successfully', result);
    } catch (err) {
      return createResponse(res, 500, err.message);
    }
  }
}

module.exports = SteganoController;
