const mongoose = require('mongoose');

const stegoImageSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },

  originalImage: {
    filename: String,
    contentType: String,
    size: Number,
    width: Number,
    height: Number,
    hash: String 
  },

  stegoDetails: {
    hasHiddenContent: {
      type: Boolean,
      default: false
    },
    
    messageLength: {
      type: Number,
      default: 0
    },
    
    isPasswordProtected: {
      type: Boolean,
      default: true
    },
    
    entropyScore: Number,
    
    usedAreas: [{
      x: Number,
      y: Number,
      width: Number,
      height: Number
    }]
  },

  watermark: {
    hasWatermark: {
      type: Boolean,
      default: false
    },
    
    watermarkType: {
      type: String,
      enum: ['visible', 'invisible', 'none'],
      default: 'none'
    },
    
    timestamp: Date
  },

  qrCode: {
    hasQRCode: {
      type: Boolean,
      default: false
    },
    
    position: {
      x: Number,
      y: Number
    }
  },

  processingDetails: {
    processedAt: {
      type: Date,
      default: Date.now
    },
    
    processingTime: Number,
    
    stegoMethod: {
      type: String,
      enum: ['lsb', 'dct', 'dwt', 'other'],
      default: 'lsb'
    },
    
    usedML: {
      type: Boolean,
      default: false
    }
  },

  label: String,
  
  tags: [String],
  
  isPublic: {
    type: Boolean,
    default: false
  },
    selfDestruct: {
    type: {
      mode: {
        type: String,
        enum: ['none', 'time', 'views'],
        default: 'none'
      },
      ttl: Number,
      expiry: Date,
    },
    default: { mode: 'none' }
  }
}, { timestamps: true });

stegoImageSchema.index({ createdAt: -1 });
stegoImageSchema.index({ 'stegoDetails.hasHiddenContent': 1 });
stegoImageSchema.index({ 'watermark.hasWatermark': 1 });
stegoImageSchema.index({ isPublic: 1 });


const StegoImage = mongoose.model('StegoImage', stegoImageSchema);
module.exports = StegoImage;                