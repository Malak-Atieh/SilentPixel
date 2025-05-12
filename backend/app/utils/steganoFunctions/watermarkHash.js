import BinaryConverter from './binaryConverter';
class WatermarkHashEmbedder {
  static store(ctx, hash, width, height) { 
       //first 16 chars of hash (64 bits) split across 4 corners
    const hashChunks = [
      hash.substring(0, 4),
      hash.substring(4, 8),
      hash.substring(8, 12),
      hash.substring(12, 16)
    ];
    
    // corner areas (3x3 pixels)
    const corners = [
      { x: 0, y: 0 },                  
      { x: width - 3, y: 0 },        
      { x: 0, y: height - 3 },         
      { x: width - 3, y: height - 3 }  
    ];
    
    // Store hash chunks in corners
    for (let i = 0; i < 4; i++) {
      const imageData = ctx.getImageData(corners[i].x, corners[i].y, 3, 3);
      const pixels = imageData.data;
      
      // Convert hash chunk to binary
      const hashBinary = BinaryConverter.textToBinary(hashChunks[i]);
      
      // Embed in alpha channel LSBs
      for (let j = 0; j < 3 * 3; j++) {
        if (j < hashBinary.length) {
          const pixelIndex = j * 4;
          const bit = parseInt(hashBinary[j]);
          
          // Modify alpha channel LSB
          pixels[pixelIndex + 3] = (pixels[pixelIndex + 3] & 0xFE) | bit;
        }
      }
      
      ctx.putImageData(imageData, corners[i].x, corners[i].y);
    }
   }

  static retrieve(ctx, width, height) { 
    
   }
}
module.exports = WatermarkHashEmbedder;