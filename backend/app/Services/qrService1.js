const sharp = require('sharp');
const QRCode = require('qrcode');

class QRService {
  static async addQRCode(imageBuffer, data) {
    try {
      const baseImage = sharp(imageBuffer);
      const metadata = await baseImage.metadata();

      const qrSize = Math.floor(Math.min(metadata.width, metadata.height) * 0.15);
      const padding = 10;

      const qrDataUrl = await QRCode.toDataURL(JSON.stringify(data), {
        errorCorrectionLevel: 'H',
        margin: 1,
        scale: 1,
        width: qrSize,
        color: {
          dark: '#00000088', // semi-transparent black
          light: '#ffffff00' // fully transparent background
        }
      });

      const qrImageBuffer = Buffer.from(qrDataUrl.split(',')[1], 'base64');

      const composite = await baseImage
        .composite([{
          input: qrImageBuffer,
          top: metadata.height - qrSize - padding,
          left: metadata.width - qrSize - padding,
          blend: 'over'
        }])
        .png()
        .toBuffer();

      return {
        data: composite,
        region: {
          x: metadata.width - qrSize - padding,
          y: metadata.height - qrSize - padding,
          width: qrSize,
          height: qrSize
        }
      };
    } catch (error) {
      throw new Error(`Error adding QR code: ${error.message}`);
    }
  }
}

module.exports = QRService;
