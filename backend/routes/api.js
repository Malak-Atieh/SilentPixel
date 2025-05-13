const express = require('express');
const router = express.Router();
const AuthController = require('../app/Controllers/AuthController');
const SteganoController = require('../app/Controllers/SteganoController');
const { validate } = require('../app/Middlewares/validate');
const { auth } = require('../app/Middlewares/Auth');
const multer = require('multer');
const upload = multer();

// Auth Routes
router.post('/register', AuthController.register);
router.post('/login', AuthController.login);

// Steganography Routes
router.post(
  '/analyze',
  auth,
  upload.single('image'),
  SteganoController.analyzeImage
);

router.post(
  '/encode', 
  auth, 
  upload.single('image'), 
  SteganoController.encode
);

router.post(
  '/decode', 
  auth, 
  SteganoController.decode
);


module.exports = router;