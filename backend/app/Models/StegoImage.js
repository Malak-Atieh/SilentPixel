const mongoose = require("mongoose");


const stegoImage = new mongoose.Schema({
  userId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "User", 
    required: true 
  },
  originalUrl: { type: String, required: true },
  encodedUrl: { type: String, required: true },
  watermarkText: { type: String },
  generateQR: { type: Boolean, default: false },
  message: { type: String, required: true },
  type: { 
    type: String, 
    enum: ['encode', 'decode'], 
    required: true 
  },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("StegoImage", stegoImage);
