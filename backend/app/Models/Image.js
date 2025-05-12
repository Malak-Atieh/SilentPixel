const mongoose = require('mongoose');

/**
 * Schema for tracking steganography operations and image metadata
 * Does not store actual image data (stored separately or returned to user)
 */
const stegoImageSchema = new mongoose.Schema({
  // Reference to user who created/processed this image
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },

  // Original image details
  originalImage: {
    filename: String,
    contentType: String,
    size: Number,
    width: Number,
    height: Number,
    hash: String // Hash of original image for verification
  },

  // Steganography metadata
  stegoDetails: {
    // If true, this image has hidden content embedded
    hasHiddenContent: {
      type: Boolean,
      default: false
    },
    
    // Length of hidden message in bytes (0 if none)
    messageLength: {
      type: Number,
      default: 0
    },
    
    // Message was protected with password
    isPasswordProtected: {
      type: Boolean,
      default: true
    },
    
    // Entropy score (complexity) of the image (higher = better for hiding)
    entropyScore: Number,
    
    // Areas used for hiding data (coordinates)
    usedAreas: [{
      x: Number,
      y: Number,
      width: Number,
      height: Number
    }]
  },

  // Watermark information
  watermark: {
    // If true, image has a watermark
    hasWatermark: {
      type: Boolean,
      default: false
    },
    
    // Type of watermark (visible/invisible)
    watermarkType: {
      type: String,
      enum: ['visible', 'invisible', 'none'],
      default: 'none'
    },
    
    // Watermark creation timestamp
    timestamp: Date
  },

  // QR code information
  qrCode: {
    // If true, image has a QR code embedded
    hasQRCode: {
      type: Boolean,
      default: false
    },
    
    // Position of QR code
    position: {
      x: Number,
      y: Number
    }
  },

  // Processing information
  processingDetails: {
    // When the image was processed
    processedAt: {
      type: Date,
      default: Date.now
    },
    
    // Time taken to process in ms
    processingTime: Number,
    
    // Method used for steganography
    stegoMethod: {
      type: String,
      enum: ['lsb', 'dct', 'dwt', 'other'],
      default: 'lsb'
    },
    
    // If ML was used to enhance steganography
    usedML: {
      type: Boolean,
      default: false
    }
  },

  // Optional descriptive label set by user
  label: String,
  
  // Tags for organizing images
  tags: [String],
  
  // Public/private setting
  isPublic: {
    type: Boolean,
    default: false
  }
}, { timestamps: true });

// Indexes for common queries
stegoImageSchema.index({ createdAt: -1 });
stegoImageSchema.index({ 'stegoDetails.hasHiddenContent': 1 });
stegoImageSchema.index({ 'watermark.hasWatermark': 1 });
stegoImageSchema.index({ isPublic: 1 });


const StegoImage = mongoose.model('StegoImage', stegoImageSchema);
module.exports = StegoImage;                