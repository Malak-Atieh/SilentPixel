const jwt = require('jsonwebtoken');
const { AuthError } = require('../Traits/errors');

const generateToken = (userId, additionalData = {}) => {
  if (!process.env.JWT_SECRET) {
    throw new Error('JWT_SECRET environment variable is not set');
  }
  
  const payload = {
    userId,
    ...additionalData
  };
  
  const options = {
    expiresIn: process.env.JWT_EXPIRES_IN || '7d'
  };
  
  return jwt.sign(payload, process.env.JWT_SECRET, options);
};

const verifyToken = (token) => {
  if (!token) {
    throw new AuthError('No token provided');
  }
  
  try {
    return jwt.verify(token, process.env.JWT_SECRET);
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      throw new AuthError('Token has expired');
    }
    throw new AuthError('Invalid token');
  }
};


module.exports = { generateToken,
  verifyToken };
