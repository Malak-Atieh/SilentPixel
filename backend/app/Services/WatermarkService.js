const crypto = require('crypto');
const { AppError } = require('../Traits/errors');
const ImageProcessor = require('../utils/imageProcessor');
const BinaryConverter = require('../utils/steganoFunctions/binaryConverter');

class WatermarkService {
    static watermarkConfig = {
        regionSize: 32,  // Reduced from 64 to minimize space usage
        edgePadding: 5,
        channel: 'g',   // Use only green channel (less noticeable to human eye)
        hashBits: 8     // Store only first 8 chars of hash (compromise between space and security)
    };

    static async addWatermark(imageBuffer, watermarkData) {
        try {
            const { image, metadata } = await ImageProcessor.loadImage(imageBuffer);
            const imageData = await ImageProcessor.getImageData(image);
            const { data, width, height } = imageData;

            // Compact watermark format
            const watermarkString = this._createCompactWatermark(watermarkData);
            const binaryWatermark = BinaryConverter.textToBinary(watermarkString);
            
            // Calculate dynamic region size based on image dimensions
            const region = this._calculateWatermarkRegion(width, height);
            
            // Embed watermark in LSB of green channel only
            this._embedWatermark(data, width, binaryWatermark, region);

            // Store minimal verification hash in corners
            this._storeVerificationHash(data, width, height, watermarkString);

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
            const { image } = await ImageProcessor.loadImage(imageBuffer);
            const imageData = await ImageProcessor.getImageData(image);
            const { data, width, height } = imageData;

            // Retrieve verification hash first
            const storedHash = this._retrieveVerificationHash(data, width, height);
            if (!storedHash) {
                throw new AppError('No watermark found', 400);
            }

            const region = this._calculateWatermarkRegion(width, height);
            const binaryWatermark = this._extractWatermark(data, width, region);
            const watermarkString = BinaryConverter.binaryToText(binaryWatermark);

            // Verify with partial hash match
            const extractedHash = crypto.createHash('sha256')
                .update(watermarkString)
                .digest('hex')
                .substring(0, this.watermarkConfig.hashBits);

            if (extractedHash !== storedHash) {
                throw new AppError('Watermark verification failed', 400);
            }

            return this._parseCompactWatermark(watermarkString);
        } catch (error) {
            throw new AppError('Watermark extraction failed: ' + error.message, 400);
        }
    }

    static _createCompactWatermark(data) {
        return `${data.email}|${data.timestamp}`;
    }

    static _parseCompactWatermark(str) {
        const [email, timestamp] = str.split('|');
        return { email, timestamp };
    }

    static _calculateWatermarkRegion(width, height) {
        const { regionSize, edgePadding } = this.watermarkConfig;
        const safeSize = Math.min(
            regionSize,
            width - edgePadding * 2,
            height - edgePadding * 2
        );
        
        return {
            startX: width - safeSize - edgePadding,
            startY: height - safeSize - edgePadding,
            size: safeSize
        };
    }

    static _embedWatermark(data, width, binaryWatermark, region) {
        const { startX, startY, size } = region;
        let bitIndex = 0;

        for (let y = startY; y < startY + size && bitIndex < binaryWatermark.length; y++) {
            for (let x = startX; x < startX + size && bitIndex < binaryWatermark.length; x++) {
                const pixelIndex = (y * width + x) * 4;
                const bit = parseInt(binaryWatermark[bitIndex]);
                data[pixelIndex + 1] = (data[pixelIndex + 1] & 0xFE) | bit;
                bitIndex++;
            }
        }
    }

    static _extractWatermark(data, width, region) {
        const { startX, startY, size } = region;
        let binaryWatermark = '';

        for (let y = startY; y < startY + size; y++) {
            for (let x = startX; x < startX + size; x++) {
                const pixelIndex = (y * width + x) * 4;
                binaryWatermark += (data[pixelIndex + 1] & 1).toString();
            }
        }

        return binaryWatermark;
    }

    static _storeVerificationHash(data, width, height, watermarkString) {
        const hash = crypto.createHash('sha256')
            .update(watermarkString)
            .digest('hex')
            .substring(0, this.watermarkConfig.hashBits);

        const binaryHash = BinaryConverter.textToBinary(hash);
        let bitIndex = 0;

        const corners = [
            { x: 0, y: 0 },
            { x: width - 1, y: 0 },
            { x: 0, y: height - 1 },
            { x: width - 1, y: height - 1 }
        ];

        for (const corner of corners) {
            if (bitIndex >= binaryHash.length) break;
            
            const pixelIndex = (corner.y * width + corner.x) * 4;
            for (let c = 0; c < 3 && bitIndex < binaryHash.length; c++) {
                const bit = parseInt(binaryHash[bitIndex]);
                data[pixelIndex + c] = (data[pixelIndex + c] & 0xFE) | bit;
                bitIndex++;
            }
        }
    }

    static _retrieveVerificationHash(data, width, height) {
        let binaryHash = '';
        const corners = [
            { x: 0, y: 0 },
            { x: width - 1, y: 0 },
            { x: 0, y: height - 1 },
            { x: width - 1, y: height - 1 }
        ];

        for (const corner of corners) {
            const pixelIndex = (corner.y * width + corner.x) * 4;
            for (let c = 0; c < 3; c++) {
                binaryHash += (data[pixelIndex + c] & 1).toString();
            }
        }

        try {
            return BinaryConverter.binaryToText(binaryHash);
        } catch {
            return null;
        }
    }
}

module.exports = WatermarkService;