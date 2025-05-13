const QECode= require('qrcode');
const Jimp = require('jimp');
const jsQR= require('jsqr');
const {createResponse} = require('../Traits/response');
class QRService {

    static async addQRCode(imageBuffer, data) {

        try {
            const image = await Jimp.read(imageBuffer);
    
            //generate the QR code
            const qrSize = Math.min(image.width, image.height) *0.15; //15% of img lenght

            //generate the QR code with the data
            const qrDataUrl = await QRCode.toDataURL(JSON.stringify(data), {
                errorCorrectionLevel: 'H',
                margin: 1,
                scale: 1,
                width: qrSize,
            });

            // Load QR code image into Jimp
            const qrImage = await Jimp.read(Buffer.from(qrDataUrl.split(",")[1], 'base64'));

            const padding = 10;
            const x = image.bitmap.width - qrSize - padding;
            const y = image.bitmap.height - qrSize - padding;

            // Composite QR onto the original image with alpha (transparency)
            qrImage.opacity(0.8);
            image.composite(qrImage, x, y);

            const modifiedBuffer = await image.getBufferAsync(Jimp.MIME_PNG);
            return createResponse(200, 'QR code added successfully', modifiedBuffer);
        } catch (error) {
            return createResponse(500, 'Error adding QR code', error);
        }
    }

    static async extractQRCode(imageBuffer) {
        try {
            const { canvas, ctx } = await ImageProcessor.loadImageToCanvas(imageBuffer);
            const imageData = ImageProcessor.getImageData(ctx, canvas.width, canvas.height);
    
            //scan the image for QR code
            const code = jsQR(imageData.data, canvas.width, canvas.height);
            let qrData = null;
            if(code){
                //parse the QR code data
                 qrData = JSON.parse(code.data);
            }
            return createResponse(200, 'QR code extracted successfully', qrData);

        } catch (error) {
            return createResponse(500, 'Error extracting QR code', error);
        }
    }
}
module.exports = QRService;