const {createResponse} = require('../Traits/response');
const crypto = require('crypto');
const ImageProcessor = require('../utils/imageProcessor');
const BinaryConverter = require('../utils/steganoFunctions/binaryConverter');
class WatermarkService {
    static async addWatermark(imageBuffer, watermarkData) {
        try {
            const { image } = await ImageProcessor.loadImage(imageBuffer);
            
            const imageData = await ImageProcessor.getImageData(image);
            const { data, width, height } = imageData;
        
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
            
            const updatedImage = ImageProcessor.updateImage(imageData);
      
            const modifiedBuffer = await ImageProcessor.imageToBuffer({ image: updatedImage });
            return createResponse(200, 'Watermark added successfully', modifiedBuffer);
        } catch (error) {
            createResponse(500, 'Error adding watermark', error);
        }
    }

    static async extractWatermark(imageBuffer) {
    try {
      const { image } = await ImageProcessor.loadImage(imageBuffer);
      
      const imageData = await ImageProcessor.getImageData(image);
      const { data, width, height } = imageData;

       const storedHash = this._retrieveWatermarkHash(data, width, height);
        if (!storedHash) {
            throw new AppError('No watermark found', 400);
        }

        const maxLength = 128 * 8; 
      const positions = this._getWatermarkPositions(width, height, maxLength);
      
      // Extract binary watermark
      let binaryWatermark = '';
      for (let i = 0; i < maxLength; i++) {
        const pos = positions[i];
        const pixelIndex = pos * 4;
        
        // Compare red and green channels to determine bit value
        const red = data[pixelIndex];
        const green = data[pixelIndex + 1];
        
        binaryWatermark += (red > green) ? '1' : '0';
      }
      
      // Convert binary to text
      const watermarkString = BinaryConverter.binaryToText(binaryWatermark);
      
      // Verify hash
      const extractedHash = crypto
        .createHash('sha256')
        .update(watermarkString)
        .digest('hex');
      
      if (extractedHash.substring(0, 16) !== storedHash.substring(0, 16)) {
        throw new AppError('Watermark hash mismatch', 400);
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
  static _getWatermarkPositions(width, height, length) {
    const totalPixels = width * height;
    const positions = [];
    
    // Use prime numbers for position calculation
    const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    
    for (let i = 0; i < length; i++) {
      // Calculate position using a formula with primes
      const prime1 = primes[i % primes.length];
      const prime2 = primes[(i + 7) % primes.length];
      
      // Generate a position that depends on current index and primes
      const position = (prime1 * i + prime2) % totalPixels;
      
      positions.push(position);
    }
    
    return positions;
  }


  static _storeWatermarkHash(data, hash, width, height) {
    const hashChunks = [
      hash.substring(0, 4),
      hash.substring(4, 8),
      hash.substring(8, 12),
      hash.substring(12, 16)
    ];
    
    const corners = [
      { x: 0, y: 0 },
      { x: width - 3, y: 0 },
      { x: 0, y: height - 3 },
      { x: width - 3, y: height - 3 }
    ];
    
    for (let i = 0; i < 4; i++) {
      const hashBinary = BinaryConverter.textToBinary(hashChunks[i]);
      
      let bitIndex = 0;
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          if (bitIndex >= hashBinary.length) break;
          
          const x = corners[i].x + dx;
          const y = corners[i].y + dy;
          const pixelIndex = (y * width + x) * 4;
          
          const bit = parseInt(hashBinary[bitIndex]);
          data[pixelIndex + 3] = (data[pixelIndex + 3] & 0xFE) | bit; // Modify alpha channel LSB
          
          bitIndex++;
        }
      }
    }
  }


  static _retrieveWatermarkHash(data, width, height) {
    const corners = [
      { x: 0, y: 0 },
      { x: width - 3, y: 0 },
      { x: 0, y: height - 3 },
      { x: width - 3, y: height - 3 }
    ];
    
    let hashBinary = '';
    
    for (let i = 0; i < 4; i++) {
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const x = corners[i].x + dx;
          const y = corners[i].y + dy;
          const pixelIndex = (y * width + x) * 4;
          
          hashBinary += (data[pixelIndex + 3] & 1).toString(); // Read alpha channel LSB
        }
      }
    }
    
    try {
      return BinaryConverter.binaryToText(hashBinary);
    } catch (e) {
      return null;
    }
  }
}

module.exports = WatermarkService;