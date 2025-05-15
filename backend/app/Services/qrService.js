const QRCode= require('qrcode');
const ImageProcessor = require('../utils/imageProcessor');
const jsQR= require('jsqr');
const { AppError } = require('../Traits/errors');
class QRService {

    static async addQRCode(imageBuffer, data) {

        try {
            const { image, metadata } = await ImageProcessor.loadImage(imageBuffer);
      console.log("here");
            //generate the QR code
            const qrSize = Math.min(metadata.width, metadata.height) *0.15; //15% of img lenght
console.log("here1");
            //generate the QR code with the data
            const qrDataUrl = await QRCode.toDataURL(JSON.stringify(data), {
                errorCorrectionLevel: 'H',
                margin: 4,
                scale: 4,
                width: qrSize,
            });
console.log("here2");
            // Load QR code image into Jimp
            const qrImageBuffer = Buffer.from(qrDataUrl.split(',')[1], 'base64');
            const { image: qrImage } = await ImageProcessor.loadImage(qrImageBuffer);
      console.log("here3");
            const padding = 10;
            const x = metadata.bitmap.width - qrSize - padding;
            const y = metadata.bitmap.height - qrSize - padding;
console.log("here4");
            // Composite QR onto the original image with alpha (transparency)
            const modifiedImage = image.composite([
            { 
            input: await qrImage.toBuffer(), 
            gravity: 'southeast',
            blend: 'over'
            }
            ]);
      
console.log("here5");
            const modifiedBuffer = await ImageProcessor.imageToBuffer({ image: modifiedImage });
            return {
                data: modifiedBuffer,
                region: { x, y, width: qrSize, height: qrSize }
            };
        } catch (error) {
            throw new Error('Error adding QR code', 500, error);
        }
    }

    static async extractQRCode(imageBuffer) {
        try {
            // Load the image using Sharp
            const { image } = await ImageProcessor.loadImage(imageBuffer);
            console.log("here0");
            // Get raw image data
            const imageData = await ImageProcessor.getImageData(image);
            console.log("here");
            // Scan for QR code
            const code = jsQR(
                imageData.data, 
                imageData.width, 
                imageData.height, 
                { inversionAttempts: "dontInvert",
                     canOverwriteImage: true 
                 }
            );
             console.log("here2");
            if (!code) {
                return null;
            }
            qrData= JSON.parse(code.data);
             console.log("here3",qrData);
            return qrrData;

        } catch (error) {
            throw new Error(500, 'Error extracting QR code', error);
        }
    }
}
module.exports = QRService;