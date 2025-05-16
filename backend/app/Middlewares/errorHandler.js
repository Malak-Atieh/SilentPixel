const { AppError } = require('../Traits/errors');

const errorHandler = (err, req, res, next) => {
  // 1. Ensure err is always an object
  if (!err || typeof err !== 'object') {
    console.error('Non-object error received:', err);
    err = new Error('Unknown error occurred');
  }

  // 2. Safely extract error details with defaults
  const statusCode = err.status || err.statusCode || 500;
  let message = err.message || 'Something went wrong';
  let errorType = err instanceof AppError ? err.name : 'ServerError';

  // 3. Enhanced error type handling
  const handleErrorType = () => {
    // Mongoose validation errors
    if (err.name === 'ValidationError' && err.errors) {
      message = Object.values(err.errors)
        .map(e => e.message || 'Invalid field')
        .join(', ');
      return 'ValidationError';
    }

    // MongoDB duplicate key
    if (err.code === 11000) {
      const field = err.keyValue ? Object.keys(err.keyValue)[0] : 'field';
      message = `Duplicate value for ${field}`;
      return 'DuplicateError';
    }

    // JWT errors
    if (err.name === 'JsonWebTokenError') {
      message = 'Invalid token';
      return 'AuthError';
    }

    // Cast errors (invalid ObjectId, etc)
    if (err.name === 'CastError') {
      message = `Invalid ${err.path}: ${err.value}`;
      return 'ValidationError';
    }

    return errorType;
  };

  errorType = handleErrorType();

  // 4. Development vs production logging
  if (process.env.NODE_ENV === 'development') {
    console.error('\x1b[31m%s\x1b[0m', `[${new Date().toISOString()}] Error:`, {
      message,
      stack: err.stack,
      originalError: err
    });
  } else {
    console.error(`[${new Date().toISOString()}] Error:`, message);
  }

  // 5. Response construction with optional stack trace
  const response = {
    success: false,
    status: statusCode,
    message,
    type: errorType,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
  };

  // 6. Special handling for 500 errors in production
  if (statusCode === 500 && process.env.NODE_ENV !== 'development') {
    response.message = 'Internal server error';
    delete response.stack;
  }

  res.status(statusCode).json(response);
};

module.exports = errorHandler;