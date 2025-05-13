import BinaryConverter from './binaryConverter';
class WatermarkHashEmbedder {

  static async storeWatermarkHash(jimpImage, hash) {
    const { width, height } = jimpImage.bitmap;

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

          const color = jimpImage.getPixelColor(x, y);
          let rgba = Jimp.intToRGBA(color);

          const bit = parseInt(hashBinary[bitIndex]);
          rgba.a = (rgba.a & 0xFE) | bit;

          const newColor = Jimp.rgbaToInt(rgba.r, rgba.g, rgba.b, rgba.a);
          jimpImage.setPixelColor(newColor, x, y);

          bitIndex++;
        }
      }
    }
  }

  static retrieve(jimpImage) {
    const { width, height } = jimpImage.bitmap;
    const corners = [
      { x: 0, y: 0 },
      { x: width - 3, y: 0 },
      { x: 0, y: height - 3 },
      { x: width - 3, y: height - 3 }
    ];

    let hashBinary = '';

    for (let corner of corners) {
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const x = corner.x + dx;
          const y = corner.y + dy;

          const color = jimpImage.getPixelColor(x, y); // returns a number
          const alpha = (color >> 24) & 0xff;
          hashBinary += (alpha & 1).toString();
        }
      }
    }

    try {
      const hashString = BinaryConverter.binaryToText(hashBinary);
      return hashString;
    } catch (e) {
      return null;
    }
  }

}
module.exports = WatermarkHashEmbedder;