const express = require('express');
const router = express.Router();
const AuthController = require('../app/Controllers/AuthController');
const SteganoController = require('../app/Controllers/SteganoController');
const { validate } = require('../app/Middlewares/validate');
const { auth } = require('../app/Middlewares/Auth');
const multer = require('multer');
const upload = multer();

// Auth Routes
router.post('/register', validate(registerSchema), AuthController.register);
router.post('/login', validate(loginSchema), AuthController.login);

// Steganography Routes
router.post(
  '/encode', 
  auth, 
  upload.single('image'), 
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