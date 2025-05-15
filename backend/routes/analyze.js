// backend/routes/analyze.js
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');

const router = express.Router();
const upload = multer();

router.post('/', upload.single('image'), async (req, res) => {
  try {
    const form = new FormData();
    form.append('image', req.file.buffer, { filename: 'img.png' });

    const response = await axios.post('http://localhost:5001/analyze', form, {
      headers: form.getHeaders()
    });

    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(500).send('ML service failed');
  }
});

module.exports = router;
