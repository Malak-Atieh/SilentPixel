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
    const user = await User.findOne({ email });
    if (!user) throw new Error('Invalid credentials');
    
    const isMatch = await comparePassword(password, user.password);
    if (!isMatch) throw new Error('Invalid credentials');

    return user;
  }
}
module.exports = UserService;
