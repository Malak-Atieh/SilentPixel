const {createResponse} = require('../Traits/response');
const crypto = require('crypto');
const Jimp = require('jimp');
const BinaryConverter = require('../utils/steganoFunctions/binaryConverter');
class WatermarkService {
    static async addWatermark(imageBuffer, watermarkData) {
        try {
            const image = await Jimp.read(imageBuffer);
            const { data, width, height } = image.bitmap;

            //create a digest of the image data
            const watermarkString = JSON.stringify(watermarkData);
            const watermarkHash = crypto.createHash('sha256').update(watermarkString).digest('hex'); 
            
            //convert watermark data to binary(using 128 of hash)
            const binaryWatermark = BinaryConverter.textToBinary(watermarkString.substring(0, 128));

            //determine watermark position
            const position = this._getWatermarkPosition(width, height, binaryWatermark.length);

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
            //store the watermark hash in the alpha channel corners
            this._storeWatermarkHash(data, watermarkHash, width, height);

            //convert the canvas to a buffer
            const modifiedBuffer = await image.getBufferAsync(Jimp.MIME_PNG);
            return createResponse(200, 'Watermark added successfully', modifiedBuffer);
        } catch (error) {
            createResponse(500, 'Error adding watermark', error);
        }
    }

    static async extractWatermark(imageBuffer) {
    try {
        const image = await Jimp.read(imageBuffer);
        const { width, height, data } = image.bitmap;

        // Retrieve watermark hash
        const storedHash = this.retrieve(image);
        if (!storedHash) {
        return createResponse(400, 'No watermark found', null);
        }

        const maxLength = 128 * 8;
        const positions = this._getWatermarkPosition(width, height, maxLength);

        let binaryWatermark = '';
        for (let pos of positions) {
        const x = pos % width;
        const y = Math.floor(pos / width);

        const color = image.getPixelColor(x, y);
        const red = (color >> 16) & 0xff;
        const green = (color >> 8) & 0xff;

        binaryWatermark += (red > green) ? '1' : '0';
        }

        const watermarkString = BinaryConverter.binaryToText(binaryWatermark);

        const extractedHash = crypto.createHash('sha256').update(watermarkString).digest('hex');
        if (extractedHash.substring(0, 16) !== storedHash.substring(0, 16)) {
        return createResponse(400, 'Watermark hash mismatch', null);
        }

        try {
        const watermarkData = JSON.parse(watermarkString);
        return createResponse(200, 'Watermark extracted successfully', watermarkData);
        } catch (e) {
        return createResponse(500, 'Error parsing watermark data', e);
        }

    } catch (error) {
        return createResponse(500, 'Error extracting watermark', error);
    }
    }

}