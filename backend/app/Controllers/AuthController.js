const { registerUser, authenticateUser } = require('../Services/UserService');
const { generateToken } = require('../utils/jwt');
const { createResponse } = require('../utils/apiResponse');

class AuthController {
  static async register(req, res) {
    try {
      const user = await registerUser(req.body);
      const token = generateToken(user._id);
      return createResponse(res, 201, 'User registered', { user, token });
    } catch (err) {
      return createResponse(res, 400, err.message);
    }
  }

  static async login(req, res) {
    try {
      const user = await authenticateUser(req.body.email, req.body.password);
      const token = generateToken(user._id);
      return createResponse(res, 200, 'Login successful', { user, token });
    } catch (err) {
      return createResponse(res, 401, err.message);
    }
  }
}

module.exports = AuthController;
