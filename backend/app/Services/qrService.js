const QECode= require('qrcode');
const {createCanvas, loadImage} = require('canvas');
const jsQR= require('jsqr');
const {createResponse} = require('../Traits/response');
class QRService {

    static async addQRCode(imageBuffer, data) {

        try {
            //load the image
            const image =loadImage(imageBuffer);

            //create a canvas
            const canvas = createCanvas(image.width, image.height);
            const ctx = canvas.getContext('2d');

            //draw the image on the canvas
            ctx.drawImage(image, 0, 0);

            //generate the QR code
            const qrSize = Math.min(image.width, image.height) *0.15; //15% of img lenght
            const qrCanvas = createCanvas(qrSize, qrSize);

            //generate the QR code with the data
            await this.addQRCode.toCanvas(qrCanvas, JSON.stringify(data), {
                errorCorrectionLevel: 'H',
                margin: 1,
                scale: 1,
                width: qrSize,
            });

            //position the QR code on the image to the bottom right corner
            const padding = 10;
            const qrPosition = {
                x: image.width - qrSize - padding,
                y: image.height - qrSize - padding,
            }

            //draw the QR code on the canvas with semi-transparent background
            ctx.globalAlpha = 0.8;
            ctx.drawImage(qrCanvas, qrPosition.x, qrPosition.y);
            ctx.globalAlpha = 1;

            //convert the canvas to a buffer
            const modifiedBuffer = canvas.toBuffer('image/png');

            return createResponse(200, 'QR code added successfully', modifiedBuffer);
        } catch (error) {
            return createResponse(500, 'Error adding QR code', error);
        }
    }


}
module.exports = QRService;