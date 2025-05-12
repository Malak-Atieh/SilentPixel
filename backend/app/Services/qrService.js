const QECode= require('qrcode');
const {createCanvas, loadImage} = require('canvas');
const jsQR= require('jsqr');

class QRService {

    static async addQRCode(imageBuffer, data) {
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

        
    }
}