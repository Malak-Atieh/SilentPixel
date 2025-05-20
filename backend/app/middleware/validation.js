const { ValidationError } = require('../Traits/errors');

const validateImageUpload = (req, res, next) => {
  if (!req.file) {
    throw new ValidationError('Image file is required');
  }
    
  const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/jpg'];
  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    throw new ValidationError('File must be a valid image (JPEG, PNG, or JPG)');
  }

  const maxSize = 10 * 1024 * 1024; 
  if (req.file.size > maxSize) {
    throw new ValidationError('File size must be less than 10MB');
  }
  
  next();
}

const validateMessageInput = (req, res, next) => {
  const { message, messages } = req.body;
  
  if (!message && !messages) {
    throw new ValidationError('Message is required');
  }

  
  next();
};

module.exports = { 
  validateImageUpload,
  validateMessageInput 
};
  