const Jimp = require('jimp');

class ImageProcessor {
  static async loadImageToCanvas(imageBuffer) {
    const image = await Jimp.read(imageBuffer);
    return { image };
  }

  static getImageData(image) {
    return {
      data: image.bitmap.data,
      width: image.bitmap.width,
      height: image.bitmap.height
    };
  }

  static async canvasToBuffer({ image }, mime = Jimp.MIME_PNG) {
    return await image.getBufferAsync(mime);  
  }
}

module.exports = ImageProcessor;