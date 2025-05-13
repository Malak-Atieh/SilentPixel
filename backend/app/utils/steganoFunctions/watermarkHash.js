import BinaryConverter from './binaryConverter';
class WatermarkHashEmbedder {
  static store(ctx, hash, width, height) { 
    //first 16 chars of hash (64 bits) split across 4 corners
    for (let i = 0; i < 16; i++) {
      const x = i % 4;
      const y = Math.floor(i / 4);
      const idx = (y * width + x) * 4 + 3; // Alpha channel
      data[idx] = hash.charCodeAt(i) & 0xFF;
    }
   }

  static retrieve(ctx, width, height) { 
        // Define corner areas (3x3 pixels)
        const corners = [
        { x: 0, y: 0 },                
        { x: width - 3, y: 0 },          
        { x: 0, y: height - 3 },        
        { x: width - 3, y: height - 3 }  
        ];
        
        let hashBinary = '';
        
        // Extract hash chunks from corners
        for (let i = 0; i < 4; i++) {
        const imageData = ctx.getImageData(corners[i].x, corners[i].y, 3, 3);
        const pixels = imageData.data;
        
        // Extract from alpha channel LSBs
        for (let j = 0; j < 3 * 3; j++) {
            const pixelIndex = j * 4;
            hashBinary += (pixels[pixelIndex + 3] & 1).toString();
        }
        }
        
        // Convert binary to string
        try {
        const hashString = BinaryConverter.binaryToText(hashBinary);
        return hashString;
        } catch (e) {
        return null;
        }
   }
}
module.exports = WatermarkHashEmbedder;