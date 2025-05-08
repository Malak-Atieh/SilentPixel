class ValidationService {

  
    static verifyMessageHash(message, password, expectedHash) {
      const calculatedHash = EncryptionService.generateHash(message, password);
      if (calculatedHash !== expectedHash) {
        throw new Error('Message integrity check failed. The image may have been tampered with.');
      }
    }
  }
  
  module.exports = ValidationService;