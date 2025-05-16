const crypto = require('crypto');
const { AppError } = require('../Traits/errors');
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
            
            const qrSafeZone = {
                x: Math.floor(width * 0.75),
                y: Math.floor(height * 0.75),
                width: Math.floor(width * 0.25),
                height: Math.floor(height * 0.25)
            };

            //determine watermark position
            const position = this._getWatermarkPositions(width, height, binaryWatermark.length,qrSafeZone);

            // Track pixel coordinates used
            let usedCoords = [];

            //embed watermark using phase coding technique 
            for (let i=0; i< binaryWatermark.length; i++){
                const pos = position[i];
                const pixelIndex = pos * 4;

                const y = Math.floor(pos / width);
                const x = pos % width;
                usedCoords.push({ x, y });

                //modifying red & grn channels in opp dir to keep overall color
                const bit = parseInt(binaryWatermark[i]);
                if(bit==1){
                    //increase red a bit, dec grn same
                    data[pixelIndex]= Math.min(255, data[pixelIndex] + 1);    
                    data[pixelIndex + 1] = Math.max(0, data[pixelIndex + 1] - 1);
                } else {
                    //dec red a bit, increase grn same
                    data[pixelIndex] = Math.max(0, data[pixelIndex] - 1);
                    data[pixelIndex + 1] = Math.min(255, data[pixelIndex + 1] + 1);
                }
            }

            const corners = [
              { x: 0, y: 0 },
              { x: width - 3, y: 0 },
              { x: 0, y: height - 3 },
              { x: width - 3, y: height - 3 }
            ];
            corners.forEach(corner => {
              for (let dy = 0; dy < 3; dy++) {
                for (let dx = 0; dx < 3; dx++) {
                  usedCoords.push({ x: corner.x + dx, y: corner.y + dy });
                }
              }
            });

            // Compute bounding box of all used pixels
            const xs = usedCoords.map(p => p.x);
            const ys = usedCoords.map(p => p.y);
            const minX = Math.max(0, Math.min(...xs));
            const minY = Math.max(0, Math.min(...ys));
            const maxX = Math.min(width, Math.max(...xs));
            const maxY = Math.min(height, Math.max(...ys));

            const wmRegion = {
              x: minX,
              y: minY,
              width: maxX - minX + 1,
              height: maxY - minY + 1
            };

            //store the watermark hash in the alpha channel corners
            this._storeWatermarkHash(data, watermarkHash, width, height);
            
            const updatedImage = ImageProcessor.updateImage(imageData);
      
            const modifiedBuffer = await ImageProcessor.imageToBuffer({ image: updatedImage });
            return {
                data: modifiedBuffer,
                region: wmRegion
              };
        } catch (error) {
          throw new AppError('Error adding watermark ' + error.message);
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
      const qrSafeZone = {
          x: Math.floor(width * 0.75),
          y: Math.floor(height * 0.75),
          width: Math.floor(width * 0.25),
          height: Math.floor(height * 0.25)
        };
      
      const positions = this._getWatermarkPositions(width, height, maxLength,qrSafeZone);
      
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
        return watermarkData;
        } catch (e) {
          throw new Error('Error parsing watermark data ' + e.message);
        }

    } catch (error) {
      throw new Error('Error extracting watermark: ' + error.message);
    }
    }
  static _getWatermarkPositions(width, height, length, qrSafeZone) {
    const totalPixels = width * height;
    const positions = [];
    
    // Use prime numbers for position calculation
    const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
        let i = 0;
        let count = 0;
        
        while (count < length) {
            // Calculate position using a formula with primes
            const prime1 = primes[i % primes.length];
            const prime2 = primes[(i + 7) % primes.length];
            
            // Generate a position that depends on current index and primes
            const position = (prime1 * i + prime2) % totalPixels;
            
            // Convert position to x,y coordinates
            const y = Math.floor(position / width);
            const x = position % width;
            
            // Skip if this position is in the QR safe zone
            if (qrSafeZone && 
                x >= qrSafeZone.x && 
                x < qrSafeZone.x + qrSafeZone.width && 
                y >= qrSafeZone.y && 
                y < qrSafeZone.y + qrSafeZone.height) {
                i++;
                continue;
            }
            
            positions.push(position);
            count++;
            i++;
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