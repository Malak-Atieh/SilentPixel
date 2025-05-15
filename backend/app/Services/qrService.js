const QRCode= require('qrcode');
const ImageProcessor = require('../utils/ImageProcessor');
const jsQR= require('jsqr');
const {createResponse} = require('../Traits/response');
class QRService {

    static async addQRCode(imageBuffer, data) {

        try {
            const { image, metadata } = await ImageProcessor.loadImage(imageBuffer);
      
            //generate the QR code
            const qrSize = Math.min(metadata.width, metadata.height) *0.15; //15% of img lenght

            //generate the QR code with the data
            const qrDataUrl = await QRCode.toDataURL(JSON.stringify(data), {
                errorCorrectionLevel: 'H',
                margin: 1,
                scale: 1,
                width: qrSize,
            });

            // Load QR code image into Jimp
            const qrImageBuffer = Buffer.from(qrDataUrl.split(',')[1], 'base64');
            const { image: qrImage } = await ImageProcessor.loadImage(qrImageBuffer);
      
            const padding = 10;
            const x = metadata.bitmap.width - qrSize - padding;
            const y = metadata.bitmap.height - qrSize - padding;

            // Composite QR onto the original image with alpha (transparency)
            const modifiedImage = image.composite([
            { 
            input: await qrImage.toBuffer(), 
            gravity: 'southeast',
            blend: 'over'
            }
            ]);
      

            const modifiedBuffer = await ImageProcessor.imageToBuffer({ image: modifiedImage });
            return createResponse(200, 'QR code added successfully', modifiedBuffer);
        } catch (error) {
            return createResponse(500, 'Error adding QR code', error);
        }
    }

    static async extractQRCode(imageBuffer) {
        try {
            // Load the image using Sharp
            const { image } = await ImageProcessor.loadImage(imageBuffer);
            
            // Get raw image data
            const imageData = await ImageProcessor.getImageData(image);
            
            // Scan for QR code
            const code = jsQR(
                imageData.data, 
                imageData.width, 
                imageData.height, 
                { inversionAttempts: "dontInvert" }
            );
            
            if (!code) {
                return null;
            }
            qrData= JSON.parse(code.data);
            return createResponse(200, 'QR code extracted successfully', qrData);

        } catch (error) {
            return createResponse(500, 'Error extracting QR code', error);
        }
    }
}
module.exports = QRService;