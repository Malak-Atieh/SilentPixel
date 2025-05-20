require('dotenv').config();
const crypto = require('crypto');
const { AppError } = require('../Traits/errors');
class EncryptionService {

  static encrypt(message, password) {
    try{
      const salt = crypto.randomBytes(process.env.SALT_LENGTH);
      const iv = crypto.randomBytes(process.env.IV_LENGTH);
      const key = crypto.scryptSync(password, salt, process.env.KEY_LENGTH, process.env.SCRYPT_PARAMS);
      const cipher = crypto.createCipheriv(process.env.ALGORITHM, key, iv);
      
      let encrypted = cipher.update(message, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      return salt.toString('hex') + iv.toString('hex') + encrypted;
    } catch (error) {
      throw new AppError(`Encryption failed: ${error.message}`, 500);
    }   
  }

  static decrypt(encryptedMsg, password) {
    if (!encryptedMsg || !password) {
      throw new AppError('Encrypted message and password are required', 400);
    }
    try {
      if (encryptedMsg.length < 64) {
        throw new Error('Invalid encrypted message format');
      }
      const salt = Buffer.from(encryptedMsg.substring(0, 32), 'hex');
      const iv = Buffer.from(encryptedMsg.substring(32, 64), 'hex');
      const ciphertext = encryptedMsg.substring(64);

      const key = crypto.scryptSync(password, salt, process.env.KEY_LENGTH, process.env.SCRYPT_PARAMS);

      const decipher = crypto.createDecipheriv(process.env.ALGORITHM, key, iv);

      let decrypted = decipher.update(ciphertext, 'hex', 'utf8');

      decrypted += decipher.final('utf8');

      return decrypted;
    } catch (error) {
      throw new AppError('Decryption failed. Invalid message or password.', 400);
    }
  }


  static generateHash(message, password) {
    return crypto.createHash('sha256')
      .update(message + password + crypto.randomBytes(16).toString('hex'))
      .digest('hex');
  }
}

module.exports = EncryptionService;