const sharp = require('sharp');
const jpeg = require('jpeg-js');
const ReedSolomon = require('@ronomon/reed-solomon');
const EncryptionService = require('../../Services/encryptionService');
const BinaryConverter = require('./binaryConverter');

class DCTSteganography {
  static QUANT_TABLE = [
    1,  1,  1,  2,  3,  5,  6,  8,
    1,  1,  2,  3,  4,  6,  8,  10,
    1,  2,  3,  4,  6,  8,  10, 12,
    2,  3,  4,  6,  8,  10, 12, 15,
    3,  4,  6,  8,  10, 12, 15, 20,
    5,  6,  8,  10, 12, 15, 20, 25,
    6,  8,  10, 12, 15, 20, 25, 30,
    8,  10, 12, 15, 20, 25, 30, 40
  ];

  static async embed(imageBuffer, message, password) {
    try {
      const jpegBuffer = await sharp(imageBuffer)
        .jpeg({ quality: 100, chromaSubsampling: '4:4:4' })
        .toColourspace('ycbcr')
        .toBuffer();

      const rawImageData = jpeg.decode(jpegBuffer);
      const { width, height } = rawImageData;

      const yChannel = this._extractChannel(rawImageData.data, width, height, 0);
      const blocks = this._extractBlocks(yChannel, width, height);

      const header = 'STEGO' + BinaryConverter.numberToBinary(message.length, 16);
      const fullMessage = header + BinaryConverter.textToBinary(message);
      
      const encoded = await this._simpleECC(fullMessage);

      this._embedData(blocks, encoded);

      const modifiedY = this._reconstructChannel(blocks, width, height);
      this._replaceChannel(rawImageData.data, modifiedY, 0);

      return jpeg.encode(rawImageData, 100).data;
    } catch (err) {
      throw new Error(`Embedding failed: ${err.message}`);
    }
  }

  static async extract(imageBuffer, password) {
    try {
      const jpegBuffer = await sharp(imageBuffer).jpeg().toBuffer();
      const rawImageData = jpeg.decode(jpegBuffer);
      const { width, height } = rawImageData;

      const yChannel = this._extractChannel(rawImageData.data, width, height, 0);
      const blocks = this._extractBlocks(yChannel, width, height);

      const extractedBinary = this._simpleDecode(this._extractData(blocks));
      
      if (!extractedBinary.startsWith('STEGO')) {
        throw new Error('No steganographic data found');
      }

      const length = parseInt(extractedBinary.substr(5, 16), 2);
      const messageBinary = extractedBinary.substr(21, length);
      
      const message = BinaryConverter.binaryToText(messageBinary);
      
      if (password) {
        try {
          return EncryptionService.decrypt(message, password);
        } catch (decryptErr) {
          if (decryptErr.message.includes('decryption')) {
            throw new Error('Incorrect password');
          }
          throw new Error('Data corrupted during transfer');
        }
      }
      
      return message;
    } catch (err) {
      throw new Error(`Extraction failed: ${err.message}`);
    }
  }

  static _embedData(blocks, binary) {
    const robustCoefficients = [3, 10, 17, 24, 31, 38, 45, 52]; 
    let bitPos = 0;

    blocks.forEach(block => {
      robustCoefficients.forEach(coeffIdx => {
        if (bitPos >= binary.length) return;
        
        const currentVal = block[coeffIdx];
        if (Math.abs(currentVal) > 3) { 
          block[coeffIdx] = (currentVal & ~1) | parseInt(binary[bitPos++]);
        }
      });
    });
  }

  static _extractData(blocks) {
    const robustCoefficients = [3, 10, 17, 24, 31, 38, 45, 52];
    let binary = '';

    blocks.forEach(block => {
      robustCoefficients.forEach(coeffIdx => {
        binary += (block[coeffIdx] & 1).toString();
      });
    });

    return binary;
  }

    static _simpleECC(data) {
        return data.split('').map(bit => bit.repeat(3)).join('');
    }

    static _simpleDecode(data) {
        let result = '';
        for (let i = 0; i < data.length; i += 3) {
            const bits = data.substr(i, 3);
            result += bits[0] === bits[1] ? bits[0] : bits[2];
        }
        return result;
    }


  static _extractBlocks(channel, width, height) {
    const blocks = [];
    const blockSize = 8;

    for (let y = 0; y < height; y += blockSize) {
      for (let x = 0; x < width; x += blockSize) {
        const block = new Array(blockSize * blockSize);

        for (let by = 0; by < blockSize; by++) {
          for (let bx = 0; bx < blockSize; bx++) {
            const py = Math.min(y + by, height - 1);
            const px = Math.min(x + bx, width - 1);
            block[by * blockSize + bx] = channel[py * width + px];
          }
        }

        const dctBlock = this._dct2d(block);
        const quantized = this._quantize(dctBlock);
        blocks.push(quantized);
      }
    }

    return blocks;
  }

  static _dct2d(matrix) {
    const N = 8;
    const output = new Array(N * N).fill(0);
    
    const cosines = Array.from({length: N}, (_, u) => 
      Array.from({length: N}, (_, x) => 
        Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N))
    ));

    for (let u = 0; u < N; u++) {
      for (let v = 0; v < N; v++) {
        let sum = 0;
        
        for (let x = 0; x < N; x++) {
          for (let y = 0; y < N; y++) {
            sum += matrix[x * N + y] * cosines[u][x] * cosines[v][y];
          }
        }
        
        const cu = u === 0 ? 1/Math.sqrt(2) : 1;
        const cv = v === 0 ? 1/Math.sqrt(2) : 1;
        output[u * N + v] = 0.25 * cu * cv * sum;
      }
    }
    
    return output;
  }

  static _idct2d(matrix) {
    const N = 8;
    const output = new Array(N * N).fill(0);
    
    const cosines = Array.from({length: N}, (_, x) => 
      Array.from({length: N}, (_, u) => 
        Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N))
      )
    );

    for (let x = 0; x < N; x++) {
      for (let y = 0; y < N; y++) {
        let sum = 0;
        
        for (let u = 0; u < N; u++) {
          for (let v = 0; v < N; v++) {
            const cu = u === 0 ? 1/Math.sqrt(2) : 1;
            const cv = v === 0 ? 1/Math.sqrt(2) : 1;
            sum += cu * cv * matrix[u * N + v] * cosines[x][u] * cosines[y][v];
          }
        }
        
        output[x * N + y] = Math.min(255, Math.max(0, Math.round(sum / 4 + 128)));
      }
    }
    
    return output;
  }


  static _embedInDCT(blocks, binaryMsg) {
    const targetIndices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let bitIndex = 0;

    const lengthBinary = BinaryConverter.numberToBinary(binaryMsg.length, 16);
    this._embedBits(blocks, 0, lengthBinary, targetIndices);
    bitIndex += lengthBinary.length;

    this._embedBits(blocks, bitIndex, binaryMsg, targetIndices);
  }

  static _embedBits(blocks, startBit, bits, indices) {
    let blockIdx = Math.floor(startBit / indices.length);
    let coeffIdx = startBit % indices.length;

    for (let i = 0; i < bits.length; i++) {
      if (blockIdx >= blocks.length) break;

      const bit = parseInt(bits[i]);
      const block = blocks[blockIdx];
      const idx = indices[coeffIdx];

      if (Math.abs(block[idx]) > 2) {
        block[idx] = (block[idx] & ~1) | bit;
      }

      coeffIdx = (coeffIdx + 1) % indices.length;
      if (coeffIdx === 0) blockIdx++;
    }
  }

  static _extractFromDCT(blocks) {
    const targetIndices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let bits = '';

    const lengthBinary = this._extractBits(blocks, 0, 16, targetIndices);
    const length = parseInt(lengthBinary, 2);

    return this._extractBits(blocks, 16, length, targetIndices);
  }

  static _extractBits(blocks, startBit, numBits, indices) {
    let bits = '';
    let blockIdx = Math.floor(startBit / indices.length);
    let coeffIdx = startBit % indices.length;

    for (let i = 0; i < numBits; i++) {
      if (blockIdx >= blocks.length) break;

      const block = blocks[blockIdx];
      const idx = indices[coeffIdx];
      bits += (block[idx] & 1).toString();

      coeffIdx = (coeffIdx + 1) % indices.length;
      if (coeffIdx === 0) blockIdx++;
    }

    return bits;
  }


  static _addErrorCorrection(binary) {
    return binary.split('').map(bit => bit.repeat(3)).join('');
  }

  static _correctErrors(binary) {
    let corrected = '';
    for (let i = 0; i < binary.length; i += 3) {
      const chunk = binary.substr(i, 3);
      corrected += chunk[0] === chunk[1] ? chunk[0] : chunk[2];
    }
    return corrected;
  }

  static _extractChannel(data, width, height, offset) {
    const channel = new Uint8Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
      channel[Math.floor(i / 4)] = data[i + offset];
    }
    return channel;
  }

  static _replaceChannel(data, channel, offset) {
    for (let i = 0; i < channel.length; i++) {
      data[i * 4 + offset] = channel[i];
    }
  }

    static _extractBlocks(channel, width, height) {
    const blocks = [];
    const blockSize = 8;

    for (let y = 0; y < height; y += blockSize) {
      for (let x = 0; x < width; x += blockSize) {
        const block = new Array(blockSize * blockSize);

        for (let by = 0; by < blockSize; by++) {
          for (let bx = 0; bx < blockSize; bx++) {
            const py = Math.min(y + by, height - 1);
            const px = Math.min(x + bx, width - 1);
            block[by * blockSize + bx] = channel[py * width + px] - 128; // Center around 0
          }
        }

        const dctBlock = this._dct2d(block);
        const quantized = dctBlock.map((val, idx) => 
          Math.round(val / this.QUANT_TABLE[idx])
        );
        blocks.push(quantized);
      }
    }

    return blocks;
  }
  static _reconstructBlocks(blocks) {
    return blocks.map(block => {
      const dequantized = block.map((val, idx) => 
        val * this.QUANT_TABLE[idx]
      );
      return this._idct2d(dequantized);
    });
  }
    static _reconstructChannel(blocks, width, height) {
    const channel = new Uint8Array(width * height);
    const blockSize = 8;
    let blockIdx = 0;

    for (let y = 0; y < height; y += blockSize) {
      for (let x = 0; x < width; x += blockSize) {
        const block = blocks[blockIdx++];
        const pixels = this._idct2d(block);

        for (let by = 0; by < blockSize; by++) {
          for (let bx = 0; bx < blockSize; bx++) {
            const py = Math.min(y + by, height - 1);
            const px = Math.min(x + bx, width - 1);
            channel[py * width + px] = pixels[by * blockSize + bx];
          }
        }
      }
    }

    return channel;
  }
}

module.exports = DCTSteganography;