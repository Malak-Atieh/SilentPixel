class BinaryConverter {
    static textToBinary(text) {
      return [...text].map(char => 
        char.charCodeAt(0).toString(2).padStart(8, '0')
      ).join('');
    }
  
    static binaryToText(binary) {
      return binary.match(/.{1,8}/g)
        .map(byte => String.fromCharCode(parseInt(byte, 2)))
        .join('');
    }
  
    static numberToBinary(num, length) {
      return num.toString(2).padStart(length, '0');
    }
  }
  
  module.exports = BinaryConverter;