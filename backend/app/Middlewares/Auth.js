const jwt = require('jsonwebtoken');
const {AuthError } = require('../Traits/AuthError');

const auth = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    throw new AuthError('Access denied. No token provided.');
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded; 
    next();
  } catch (err) {
    throw new AuthError('Invalid token.');
  }
};

module.exports = { auth };
