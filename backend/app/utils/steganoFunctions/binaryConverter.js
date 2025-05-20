class BinaryConverter {
    static textToBinary(text) {
      const result = new Array(text.length);
      for (let i = 0; i < text.length; i++) {
          result[i] = text.charCodeAt(i).toString(2).padStart(8, '0');
        }
      return result.join('');
    }
  
    static binaryToText(binary) {
      const chunkSize = 8 * 1024; 
      let result = '';
      
      for (let i = 0; i < binary.length; i += chunkSize) {
        const chunk = binary.substring(i, Math.min(i + chunkSize, binary.length));
        const bytes = chunk.match(/.{1,8}/g) || [];
        
        for (let j = 0; j < bytes.length; j++) {
          result += String.fromCharCode(parseInt(bytes[j], 2));
        }
      }
      
      return result;
    }
  
    static numberToBinary(num, length) {
      return num.toString(2).padStart(length, '0');
    }
  }
  
  module.exports = BinaryConverter;