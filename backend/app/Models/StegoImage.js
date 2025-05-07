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
    default: false 
  },
  hasQeCode: { 
    type: Boolean, 
    default: false 
  },
  busyAreasMap: {
    type: Object,
    default: null
  },
  type: { 
    type: String, 
    enum: ['encode', 'decode'], 
    required: true 
  },
  metaData: {
    encodingMethod: String,
    messageLength: Number,
    watermarkInfo: String,
    qrCodeData: String
  },
  createdAt: { 
    type: Date, 
    default: Date.now 
  }
});

module.exports = mongoose.model("StegoImage", stegoImageSchema);
