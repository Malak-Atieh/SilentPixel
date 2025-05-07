const mongoose = require("mongoose");


const stegoImageSchema = new mongoose.Schema({
  userId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: "User", 
    required: true 
  },
  originalImagePath: { 
    type: String, 
    required: true 
  },
  encodedImagePath: { 
    type: String, 
    required: false 
  },
  hasWatermark: { 
    type: Boolean,
    required: false 
  },
  generateQR: { type: Boolean, default: false },
  message: { type: String, required: true },
  type: { 
    type: String, 
    enum: ['encode', 'decode'], 
    required: true 
  },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("StegoImage", stegoImageSchema);
