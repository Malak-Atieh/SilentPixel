const {createCanvas, loadImage} = require ('canvas');
const {createResponse} = require('../Traits/response');
const crypto = require('crypto');

class WatermarkService {
    static async addWatermark(imageBuffer, watermarkData) {
        //load the image
        const image = await loadImage(imageBuffer);

        //create a canvas
        const canvas = createCanvas(image.width, image.height);
        const ctx = canvas.getContext('2d');

        //draw the image on the canvas
        ctx.drawImage(image, 0, 0);

        //get image data 
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;

        //create a digest of the image data
        
    }
}