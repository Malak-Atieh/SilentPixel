const crypto = require('crypto');
const { AppError } = require('Traits/errors');
class EncryptionService {

  static encrypt(message, password) {
    try{
      const salt = crypto.randomBytes(16);
      const key = crypto.scryptSync(password, salt, 32);
      const iv = crypto.randomBytes(16);
      const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
      
      let encrypted = cipher.update(message, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      return salt.toString('hex') + iv.toString('hex') + encrypted;
    } catch (error) {
      throw new AppError(`Encryption failed: ${error.message}`, 500);
    }   
  }

  static decrypt(encryptedMsg, password) {
    try {
    const salt = Buffer.from(encryptedMsg.substring(0, 32), 'hex');
    const iv = Buffer.from(encryptedMsg.substring(32, 64), 'hex');
    const ciphertext = encryptedMsg.substring(64);
    const key = crypto.scryptSync(password, salt, 32);

    const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
    let decrypted = decipher.update(ciphertext, 'hex', 'utf8');

    decrypted += decipher.final('utf8');
    
    return decrypted;
    } catch (error) {
      throw new Error('Decryption failed. Invalid message or password.');
    }
  }

  static generateHash(message, password) {
    return crypto.createHash('sha256')
      .update(message + password)
      .digest('hex');
  }
}

module.exports = EncryptionService;