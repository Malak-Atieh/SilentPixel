const express = require('express');
const router = express.Router();
const AuthController = require('../app/Controllers/AuthController');
const SteganoController = require('../app/Controllers/SteganoController');
const { validate } = require('../app/Middlewares/validate');
const { auth } = require('../app/Middlewares/auth');
const { 
  loginSchema, 
  registerSchema, 
} = require('../app/Requests/Auth');
const { 
  encodeSchema, 
  decodeSchema, 
} = require('../app/Requests');

// Auth Routes
router.post('/register', validate(registerSchema), AuthController.register);
router.post('/login', validate(loginSchema), AuthController.login);

// Steganography Routes
router.post(
  '/encode', 
  auth, 
  validate(encodeSchema), 
  SteganoController.encode
);

router.post(
  '/decode', 
  auth, 
  validate(decodeSchema), 
  SteganoController.decode
);

module.exports = router;