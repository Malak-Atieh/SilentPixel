const QRCode= require('qrcode');
const ImageProcessor = require('../utils/imageProcessor');
const jsQR= require('jsqr');
const sharp= require('sharp');
const { AppError } = require('../Traits/errors');
class QRService {

    static async addQRCode(imageBuffer, data) {

        try {
            const { metadata } = await ImageProcessor.loadImage(imageBuffer);

            const qrSize = Math.min(metadata.width, metadata.height) *0.15; //15% of img lenght

            const qrDataUrl = await QRCode.toDataURL(JSON.stringify(data), {
                errorCorrectionLevel: 'H',
                margin: 1,
                scale: 4,
                width: qrSize,
            })

            const qrImageBuffer = Buffer.from(qrDataUrl.split(',')[1], 'base64');

            const padding = 10;
            const x = metadata.width - qrSize - padding;
            const y = metadata.height - qrSize - padding;

            const modifiedImage = await sharp(imageBuffer) .composite([
            { 
                input: qrImageBuffer, 
                top: Math.round(y),
                left: Math.round(x),
            }
            ])
            .toBuffer();
      
            return {
                data: await modifiedImage,
                region: { x, y, width: qrSize, height: qrSize }
            };
        } catch (error) {
            throw new Error('Error adding QR code', 500, error);
        }
    }

    static async extractQRCode(imageBuffer) {
        try {
            const { image } = await ImageProcessor.loadImage(imageBuffer);

            const imageData = await ImageProcessor.getImageData(image);

            const code = jsQR(
                imageData.data, 
                imageData.width, 
                imageData.height, 
            );

            if (!code) {
                return null;
            }
            const qrData= JSON.parse(code.data);

            return qrData;

        } catch (error) {
            throw new Error(500, 'Error extracting QR code', error);
        }
    }
}
module.exports = QRService;