const {createCanvas, loadImage} = require ('canvas');
const {createResponse} = require('../Traits/response');
const crypto = require('crypto');
const ImageProcessor = require('../utils/steganoFunctions/imageProcessor');
const BinaryConverter = require('../utils/steganoFunctions/binaryConverter');
class WatermarkService {
    static async addWatermark(imageBuffer, watermarkData) {
        try {

            const { canvas, ctx } = await ImageProcessor.loadImageToCanvas(imageBuffer);
            const imageData = ImageProcessor.getImageData(ctx, canvas.width, canvas.height);
            
            const pixels = imageData.data;

            //create a digest of the image data
            const watermarkString = JSON.stringify(watermarkData);
            const watermarkHash = crypto.createHash('sha256').update(watermarkString).digest('hex'); 
            
            //convert watermark data to binary(using 128 of hash)
            const binaryWatermark = BinaryConverter.textToBinary(watermarkString.substring(0, 128));

            //determine watermark position
            const position = this._getWatermarkPosition(canvas.width, canvas.height, binaryWatermark.length);

            //embed watermark using phase coding technique 
            for (let i=0; i< binaryWatermark.length; i++){
                const pos = position[i];
                const pixelIndex = pos * 4;

                //modifying red & grn channels in opp dir to keep overall color
                const bit = parseInt(binaryWatermark[i]);
                if(bit==1){
                    //increase red a bit, dec grn same
                    pixels[pixelIndex]= Math.min(255, pixels[pixelIndex] + 1);    
                    pixels[pixelIndex + 1] = Math.max(0, pixels[pixelIndex + 1] - 1);
                } else {
                    //dec red a bit, increase grn same
                    pixels[pixelIndex] = Math.max(0, pixels[pixelIndex] - 1);
                    pixels[pixelIndex + 1] = Math.min(255, pixels[pixelIndex + 1] + 1);
                }
            }

            //update canvas with modified pixels
            ctx.putImageData(imageData, 0, 0);

            //store the watermark hash in the alpha channel corners
            this._storeWatermarkHash(ctx, watermarkHash, canvas.width, canvas.height);

            //convert the canvas to a buffer
            const modifiedBuffer = canvas.toBuffer('image/png');

            return createResponse(200, 'Watermark added successfully', modifiedBuffer);
        } catch (error) {
            createResponse(500, 'Error adding watermark', error);
        }
    }

    static async extractWatermark(imageBuffer) {
        try {
            const { canvas, ctx } = await ImageProcessor.loadImageToCanvas(imageBuffer);
            const imageData = ImageProcessor.getImageData(ctx, canvas.width, canvas.height);
            
            const pixels = imageData.data;

            //retrieve watermark hash from alpha channel corners
            const storedHash = this._retrieveWatermarkHash(ctx, canvas.width, canvas.height);   

            if(!storedHash) {
                return createResponse(400, 'No watermark found',null);
            }

            //max length of watermark data
            const maxLength = 128 * 8 ; 

            //get watermark position
            const positions = this._getWatermarkPosition(canvas.width, canvas.height, maxLength);

            //extract binary watermark
            let binaryWatermark = '';
            for(i=0; i<positions.length; i++){
                const pos = positions[i];
                const pixelIndex = pos * 4;

                const red= pixels[pixelIndex];
                const green= pixels[pixelIndex + 1];

                binaryWatermark += (red > green) ? '1' : '0';
            }

            //convert binary to string
            const watermarkString = BinaryConverter.binaryToText(binaryWatermark);

            //validate watermark hash
            const extractedHash = crypto.createHash('sha256').update(watermarkString).digest('hex');
            if(extractedHash.substring(0,16) !== storedHash.substring(0,16)){ 
                return createResponse(400, 'Watermark hash mismatch', null);
            }

            try{
                //parse the watermark data
                const watermarkData = JSON.parse(watermarkString);
                return createResponse(200, 'Watermark extracted successfully', watermarkData);
            } catch(e){
                return createResponse(500, 'Error parsing watermark data', e);
            }
        } catch (error) {
            return createResponse(500, 'Error extracting watermark', error);
        }
    }
}