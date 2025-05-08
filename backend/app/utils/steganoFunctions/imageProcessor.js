const { createCanvas, loadImage } = require('canvas');

class ImageProcessor {
  static async loadImageToCanvas(imageBuffer) {
    const image = await loadImage(imageBuffer);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0);
    return { canvas, ctx };
  }


}

module.exports = ImageProcessor;