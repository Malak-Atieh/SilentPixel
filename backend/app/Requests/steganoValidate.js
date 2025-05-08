class SteganoValidate {
    static validateCapacity(width, height, busyAreas, requiredBits) {
      const availablePixels = PixelSelector.getIndices(width, height, busyAreas, requiredBits);
      if (availablePixels.length < requiredBits) {
        throw new Error(`Insufficient capacity: need ${requiredBits} pixels, but only ${availablePixels.length} available.`);
      }
    }
  
    static verifyMessageHash(message, password, expectedHash) {
      const calculatedHash = EncryptionService.generateHash(message, password);
      if (calculatedHash !== expectedHash) {
        throw new Error('Message integrity check failed. The image may have been tampered with.');
      }
    }
  }
  
  module.exports = SteganoValidate;