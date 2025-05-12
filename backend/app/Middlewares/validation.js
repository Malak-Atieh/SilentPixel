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
}
  module.exports = { validate };
  