const fs = require('fs');
const path = require('path');
const SteganoService = require('../../app/services/SteganoService');
const mockCompress = (buffer) => {
    return buffer.slice(0, Math.floor(buffer.length * 0.8)); 
  };//mock compression in social media

describe('Stegano Logic Unit Tests', () => {
    let validImageBuffer;

    beforeAll(() => {
      validImageBuffer = fs.readFileSync(path.join(__dirname, '../images/test.png'));
    });

   it('Encoding returns data', async () => {
        const result = await SteganoService.encode(Buffer.from('image'), 'watermark');
        expect(result).toHaveProperty('buffer');
    });

  it('Decoding returns message', async () => {
    const encoded = await SteganoService.encode(Buffer.from('image'), 'msg');
    const message = await SteganoService.decode(encoded.buffer);
    expect(message).toBe('msg');
  });

  it('Validates URL transformation', () => {
    const filePath = 'uploads/encoded/123.png';
    const url = SteganoService.buildPublicUrl(filePath);
    expect(url).toMatch(/\/uploads\/encoded\/123.png$/);
  });

  // test if no message
  it('Fails on empty message', async () => {
    await expect(SteganoService.encode(Buffer.from('image'), ''))
      .rejects
      .toThrow('Message cannot be empty');
  });

  //test if wrong image
  it("Rejects encoding for non-image buffers", async () => {
    await expect(SteganoService.encode(Buffer.from("not-an-image"), "message"))
      .rejects
      .toThrow("Invalid image");
  });

  //test if image is corruptes dataloss  
  it("Fails when stego-image is corrupted", async () => {
    const encoded = await SteganoService.encode(validImageBuffer, "secret");
    const corruptedBuffer = encoded.buffer.slice(0, 100); 
    await expect(SteganoService.decode(corruptedBuffer))
      .rejects
      .toThrow("Corrupted data");
  });

  // test on image after social compress
  it('Extracts message after JPEG compression', async () => {
    const encoded = await SteganoService.encode(validImageBuffer, 'msg');
    const compressed = mockCompress(encoded.buffer);
    const message = await SteganoService.decode(compressed);
    expect(message).toBe('msg');
  });
});