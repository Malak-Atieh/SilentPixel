const { ValidationError } = require('../Traits/errors');
/*
const validate = (schema) => (req, res, next) => {
    const { error } = schema.validate(req.body);
    if (error) {
      return res.status(400).json({ message: error.details[0].message });
    }
    next();
  };
  */
 
const validateImageUpload = (req, res, next) => {
  if (!req.file) {
    throw new ValidationError('Image file is required');
  }
    
  const allowedMimeTypes = ['image/jpeg', 'image/png', 'image/gif'];
  if (!allowedMimeTypes.includes(req.file.mimetype)) {
    throw new ValidationError('File must be a valid image (JPEG, PNG, or GIF)');
  }

  const maxSize = 10 * 1024 * 1024; 
  if (req.file.size > maxSize) {
    throw new ValidationError('File size must be less than 10MB');
  }
  
  next();
}
  module.exports = { validate };
  