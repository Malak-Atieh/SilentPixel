require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const connectDB = require('./config/database');
const cors = require('cors');
const fileUpload = require('express-fileupload');

// Initialize app
const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(fileUpload());

// Database
connectDB();

// Routes
app.use('/api', require('./routes/api'));

// Error handling
app.use(require('./app/Utils/errorHandler'));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));