const { AppError } = require('../Traits/errors');


const errorHandler = (err, req, res, next) => {
  console.error(err);
  
  // Determine if this is a known AppError or an unknown error
  let statusCode = err.status || 500;
  let message = err.message || 'Something went wrong';
  let errorType = err instanceof AppError ? err.name : 'ServerError';
  
  // Handle Mongoose validation errors
  if (err.name === 'ValidationError') {
    statusCode = 400;
    message = Object.values(err.errors).map(e => e.message).join(', ');
    errorType = 'ValidationError';
  }
  
  // Handle MongoDB duplicate key errors
  if (err.code === 11000) {
    statusCode = 400;
    message = 'Duplicate entry found';
    errorType = 'DuplicateError';
  }
  
  // Handle JWT errors
  if (err.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Invalid token';
    errorType = 'AuthError';
  }
  
  // Create standardized error response
  const errorResponse = {
    success: false,
    status: statusCode,
    message,
    type: errorType,
  };
  
  res.status(statusCode).json(errorResponse);
};

module.exports = errorHandler;