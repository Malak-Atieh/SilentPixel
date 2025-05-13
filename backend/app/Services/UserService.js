const User = require('../Models/User');
const { hashPassword, comparePassword } = require('../utils/hash');
const { ValidationError, AuthError } = require('../Traits/errors');

class UserService {

  static async  registerUser(data) {
    try {
      if (!data.email || !data.username || !data.password) {
        throw new ValidationError('Email, username and password are required');
      }
      
      const existingUser = await User.findOne({ 
        $or: [
          { email: data.email },
          { username: data.username }
        ]
      });
      
      if (existingUser) {
        throw new ValidationError('User with this email or username already exists');
      }
      
      // Hash password and create user
      data.password = await hashPassword(data.password);
      const user = await User.create(data);
      
      // Return user without password
      const userObject = user.toObject();
      delete userObject.password;
      
      return userObject;
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new Error(`Registration failed: ${error.message}`);
    }
  }

  static async authenticateUser(email, password) {
    try {
      if (!email || !password) {
        throw new ValidationError('Email and password are required');
      }
      
      const user = await User.findOne({ email });
      if (!user) {
        throw new AuthError('Invalid credentials');
      }
      
      const isMatch = await comparePassword(password, user.password);
      if (!isMatch) {
        throw new AuthError('Invalid credentials');
      }

      const userObject = user.toObject();
      delete userObject.password;
      
      return userObject;
    } catch (error) {
      if (error instanceof AuthError || error instanceof ValidationError) {
        throw error;
      }
      throw new Error(`Authentication failed: ${error.message}`);
    }
  }

    static async getUserById(userId) {
    try {
      const user = await User.findById(userId);
      
      if (!user) {
        throw new ValidationError('User not found');
      }
      
      // Return user without password
      const userObject = user.toObject();
      delete userObject.password;
      
      return userObject;
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new Error(`Failed to get user: ${error.message}`);
    }
  }
}
module.exports = UserService;
