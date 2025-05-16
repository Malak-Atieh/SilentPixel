const crypto = require('crypto');
const { AppError } = require('../Traits/errors');
class EncryptionService {
  static ALGORITHM = 'aes-256-cbc';
  static SALT_LENGTH = 16;
  static IV_LENGTH = 16;
  static KEY_LENGTH = 32;
  static SCRYPT_PARAMS = { N: 16384, r: 8, p: 1 };

  static encrypt(message, password) {
    try{
      const salt = crypto.randomBytes(this.SALT_LENGTH);
      const iv = crypto.randomBytes(this.IV_LENGTH);
      const key = crypto.scryptSync(password, salt, this.KEY_LENGTH, this.SCRYPT_PARAMS);
      const cipher = crypto.createCipheriv(this.ALGORITHM, key, iv);
      
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

      const key = crypto.scryptSync(password, salt, this.KEY_LENGTH, this.SCRYPT_PARAMS);
    console.log('here4.3');

      const decipher = crypto.createDecipheriv(this.ALGORITHM, key, iv);
          console.log('here4.4');

      let decrypted = decipher.update(ciphertext, 'hex', 'utf8');
    console.log('here4.5');

      decrypted += decipher.final('utf8');
          console.log('here5');

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