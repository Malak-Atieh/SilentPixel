const {verifyToken} = require('../utils/jwt');
const {AuthError } = require('../Traits/errors');

const auth = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    throw new AuthError('Access denied. No token provided.');
  }

  try {
    const decoded = verifyToken(token, process.env.JWT_SECRET);
    req.user = decoded; 
    next();
  } catch (err) {
    throw new AuthError('Invalid token.');
  }
};

module.exports = { auth };
