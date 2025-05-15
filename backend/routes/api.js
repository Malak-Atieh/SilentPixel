const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const AuthController = require('../app/Controllers/AuthController');
const SteganoController = require('../app/Controllers/SteganoController');
const { auth } = require('../app/Middlewares/Auth');
const {
  validateImageUpload,
  validateMessageInput
} = require('../app/Middlewares/Validation');
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
  validateImageUpload,
  SteganoController.analyzeImage
);

router.post(
  '/encode', 
  auth, 
  upload.single('image'), 
  validateImageUpload,
  validateMessageInput,
  SteganoController.encode
);

router.post(
  '/decode', 
  auth, 
  upload.single('image'),
  validateImageUpload,
  SteganoController.decode
);
router.get('/download/:filename', (req, res) => {
  const filePath = path.join(__dirname, '../storage/uploads', req.params.filename);

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ success: false, message: 'File not found' });
  }

  res.download(filePath, err => {
    if (err) {
      return res.status(500).json({ success: false, message: 'File download failed' });
    }
  });
});
module.exports = router;