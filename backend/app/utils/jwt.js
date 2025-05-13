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


module.exports = { generateToken };
