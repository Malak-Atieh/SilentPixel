const sharp = require('sharp');

class ImageProcessor {

    static async loadImage(imageBuffer, options = { ensureAlpha: true }) {
        try {
            let image = sharp(imageBuffer);
            if (options.ensureAlpha) {
                image = image.ensureAlpha();
            }
            
            const metadata = await image.metadata();
            return { image, metadata };
        } catch (error) {
            throw new Error(`Image loading failed: ${error.message}`);
        }
    }

    static async getImageData(image) {
        try {
            const { data, info } = await image
                .raw()
                .toBuffer({ resolveWithObject: true });
            
            return {
                data: new Uint8Array(data),
                width: info.width,
                height: info.height,
                channels: info.channels
            };
        } catch (error) {
            throw new Error(`Failed to get image data: ${error.message}`);
        }
    }

    static async imageToBuffer({ image }, { mime = 'png', quality = 90 } = {}) {
        try {
            const outputOptions = {};
            if (mime === 'jpeg' || mime === 'webp') {
                outputOptions.quality = quality;
            }
            
            return await image
                [mime](outputOptions)
                .toBuffer();
        } catch (error) {
            throw new Error(`Conversion to buffer failed: ${error.message}`);
        }
    }

    static updateImage({ data, width, height, channels }) {
        return sharp(Buffer.from(data), {
            raw: { width, height, channels }
        });
    }
}

module.exports = ImageProcessor;