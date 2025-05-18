const crypto = require('crypto');
const { AppError } = require('../Traits/errors');
const ImageProcessor = require('../utils/imageProcessor');
const BinaryConverter = require('../utils/steganoFunctions/binaryConverter');

class WatermarkService {
    // Configuration for watermark placement
    static watermarkConfig = {
        regionSize: 64, // Size of the square region to use for watermark (in pixels)
        edgePadding: 10, // Padding from image edges
        channel: 'rg' // Use both red and green channels for more capacity
    };

    static async addWatermark(imageBuffer, watermarkData) {
        try {
            const { image } = await ImageProcessor.loadImage(imageBuffer);
            const imageData = await ImageProcessor.getImageData(image);
            const { data, width, height } = imageData;

            // Prepare watermark data
            const watermarkString = JSON.stringify(watermarkData);
            const watermarkHash = crypto.createHash('sha256').update(watermarkString).digest('hex');
            const binaryWatermark = BinaryConverter.textToBinary(watermarkString);

            // Calculate watermark region (bottom-right corner by default)
            const region = this._calculateWatermarkRegion(width, height);
            
            // Store watermark in the designated region
            const usedPixels = this._embedWatermarkInRegion(
                data, 
                width, 
                binaryWatermark, 
                region
            );

            // Store hash in corners for verification
            this._storeWatermarkHash(data, watermarkHash, width, height);

            const updatedImage = ImageProcessor.updateImage(imageData);
            const modifiedBuffer = await ImageProcessor.imageToBuffer({ image: updatedImage });

            return {
                data: modifiedBuffer,
                region: {
                    x: region.startX,
                    y: region.startY,
                    width: region.size,
                    height: region.size
                }
            };
        } catch (error) {
            throw new AppError('Error adding watermark: ' + error.message);
        }
    }

    static async extractWatermark(imageBuffer) {
      try {
          console.log('Starting watermark extraction...');
          const { image } = await ImageProcessor.loadImage(imageBuffer);
          const imageData = await ImageProcessor.getImageData(image);
          const { data, width, height } = imageData;
          console.log(`Image dimensions: ${width}x${height}`);

          // First verify the hash from the corners
          const storedHash = this._retrieveWatermarkHash(data, width, height);
          console.log('Stored hash (partial):', storedHash?.substring(0, 16));
          if (!storedHash) {
              throw new AppError('No watermark hash found in image corners', 400);
          }

          // Calculate where the watermark should be
          const region = this._calculateWatermarkRegion(width, height);
          console.log('Watermark region:', region);

          // Extract binary watermark from the region
          const binaryWatermark = this._extractWatermarkFromRegion(
              data, 
              width, 
              region
          );
          console.log(`Extracted ${binaryWatermark.length} bits`);

          // Convert to text
          const watermarkString = BinaryConverter.binaryToText(binaryWatermark);
          console.log('Watermark string:', watermarkString.substring(0, 50) + '...');
          
          // Verify hash
          const extractedHash = crypto
              .createHash('sha256')
              .update(watermarkString)
              .digest('hex');
          console.log('Extracted hash (partial):', extractedHash.substring(0, 16));

          if (extractedHash.substring(0, 16) !== storedHash.substring(0, 16)) {
              throw new AppError('Watermark hash mismatch', 400);
          }

          return JSON.parse(watermarkString);
      } catch (error) {
          console.error('Watermark extraction error:', error.message);
          if (error instanceof AppError) {
              throw error;
          }
          throw new AppError('Watermark extraction failed: ' + error.message, 400);
      }
  }
    static _calculateWatermarkRegion(width, height) {
        const { regionSize, edgePadding } = this.watermarkConfig;
        
        return {
            startX: width - regionSize - edgePadding,
            startY: height - regionSize - edgePadding,
            size: regionSize
        };
    }

    static _embedWatermarkInRegion(data, width, binaryWatermark, region) {
        const { startX, startY, size } = region;
        const usedPixels = [];
        let bitIndex = 0;

        for (let y = startY; y < startY + size && bitIndex < binaryWatermark.length; y++) {
            for (let x = startX; x < startX + size && bitIndex < binaryWatermark.length; x++) {
                const pixelIndex = (y * width + x) * 4;
                const bit = parseInt(binaryWatermark[bitIndex]);

                // Alternate between red and green channels
                const channel = bitIndex % 2 === 0 ? 0 : 1; // 0: red, 1: green
                
                // Store bit in LSB of the selected channel
                data[pixelIndex + channel] = (data[pixelIndex + channel] & 0xFE) | bit;
                
                usedPixels.push({ x, y });
                bitIndex++;
            }
        }

        return usedPixels;
    }

    static _extractWatermarkFromRegion(data, width, region) {
        const { startX, startY, size } = region;
        const maxBits = size * size * 2; // 2 bits per pixel (red and green)
        let binaryWatermark = '';
        let bitCount = 0;

        for (let y = startY; y < startY + size && bitCount < maxBits; y++) {
            for (let x = startX; x < startX + size && bitCount < maxBits; x++) {
                const pixelIndex = (y * width + x) * 4;
                
                // Extract from both red and green channels
                const redBit = data[pixelIndex] & 1;
                const greenBit = data[pixelIndex + 1] & 1;
                
                binaryWatermark += redBit.toString();
                binaryWatermark += greenBit.toString();
                bitCount += 2;
            }
        }

        return binaryWatermark;
    }

    static _storeWatermarkHash(data, hash, width, height) {
        // Store only first 16 chars of hash (128 bits) in alpha channels of corners
        const hashPart = hash.substring(0, 16);
        const binaryHash = BinaryConverter.textToBinary(hashPart);

        const corners = [
            { x: 0, y: 0 },               // Top-left
            { x: width - 1, y: 0 },        // Top-right
            { x: 0, y: height - 1 },       // Bottom-left
            { x: width - 1, y: height - 1 } // Bottom-right
        ];

        let bitIndex = 0;
        for (const corner of corners) {
            if (bitIndex >= binaryHash.length) break;
            
            const pixelIndex = (corner.y * width + corner.x) * 4;
            // Store 4 bits in each corner's alpha channel
            for (let i = 0; i < 4 && bitIndex < binaryHash.length; i++) {
                const bit = parseInt(binaryHash[bitIndex]);
                data[pixelIndex + 3] = (data[pixelIndex + 3] & ~(1 << i)) | (bit << i);
                bitIndex++;
            }
        }
    }

    static _retrieveWatermarkHash(data, width, height) {
        const corners = [
            { x: 0, y: 0 },
            { x: width - 1, y: 0 },
            { x: 0, y: height - 1 },
            { x: width - 1, y: height - 1 }
        ];

        let binaryHash = '';
        
        for (const corner of corners) {
            const pixelIndex = (corner.y * width + corner.x) * 4;
            // Extract 4 bits from each corner's alpha channel
            for (let i = 0; i < 4; i++) {
                binaryHash += ((data[pixelIndex + 3] >> i) & 1).toString();
            }
        }

        try {
            return BinaryConverter.binaryToText(binaryHash);
        } catch (e) {
            return null;
        }
    }
}

module.exports = WatermarkService;