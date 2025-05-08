const { createCanvas, loadImage } = require('canvas');

class ImageProcessor {
  static async loadImageToCanvas(imageBuffer) {
    const image = await loadImage(imageBuffer);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    return { canvas, ctx };
  }

  static getImageData(ctx, width, height) {
    return ctx.getImageData(0, 0, width, height);
  }

  static updateImageData(ctx, imageData) {
    ctx.putImageData(imageData, 0, 0);
  }

  static canvasToBuffer(canvas, format = 'image/png') {
    return canvas.toBuffer(format);
  }
}

module.exports = ImageProcessor;