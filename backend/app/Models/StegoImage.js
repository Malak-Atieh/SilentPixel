const mongoose = require("mongoose");


const stegoImage = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  originalUrl: { type: String, required: true },
  encodedUrl: { type: String, required: true },
  watermark: {
    text: { type: String, default: "" },
    position: { type: String, enum: ["top", "bottom", "center"], default: "bottom" }
  },
  qrCode: { type: Boolean, default: false },
  secretMessage: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("StegoImage", stegoImage);
