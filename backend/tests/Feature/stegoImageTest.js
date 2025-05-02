const path = require('path');
const fs = require('fs');
const request = require('supertest');
const app = require('../../app');
const User = require('../../app/models/User');
const StegoImage = require('../../app/models/StegoImage');

let token;
let testImageBuffer;

beforeAll(async () => {
  await User.deleteMany();
  const user = await User.create({ 
    username: 'user', 
    email: 'user@e.com', 
    password: await User.hashPassword('pass') 
  });
  const res = await request(app)
    .post('/api/login')
    .send({ email: 'user@e.com', password: 'pass' });
  token = res.body.token;
  testImageBuffer = fs.readFileSync(path.join(__dirname, '../images/test.png'));
});

afterEach(async () => {
  await StegoImage.deleteMany();
  await User.deleteMany(); 
});

describe('StegoImage Feature Tests', () => {
  
  //success watermark hide
  it('Can encode an image with watermark', async () => {
    const res = await request(app)
      .post('/api/stegano/encode')
      .set('Authorization', `Bearer ${token}`)
      .field('message', 'secret')
      .attach('image', testImageBuffer);

    expect(res.statusCode).toBe(200);
    expect(res.body).toMatchObject({
        success: true,
        data: {
          encoded_url: expect.any(String),
          watermark: 'test-watermark' 
        }
      });
  });

  //fail test on missing image to hide in
  it('Fails on missing image', async () => {
    const res = await request(app)
      .post('/api/stegano/encode')
      .set('Authorization', `Bearer ${token}`)
      .field('message', 'secret');

    expect(res.statusCode).toBe(400);
    expect(res.body.error).toBe('Image is required');
  });

  //test of authorization
  it('Requires authentication', async () => {
    const res = await request(app)
      .post('/api/stegano/encode')
      .attach('image', testImageBuffer)
      .field('message', 'secret');
    expect(res.statusCode).toBe(401);
  });

  //test on QR code extraction 
  it('Extracts hidden QR code from encoded image', async () => {
    const encodeRes = await request(app)
      .post('/api/stegano/encode')
      .set('Authorization', `Bearer ${token}`)
      .field('message', 'secret')
      .field('qrCode', 'true') // Enable QR
      .attach('image', testImageBuffer);

    const decodeRes = await request(app)
      .post('/api/stegano/decode')
      .set('Authorization', `Bearer ${token}`)
      .attach('image', encodeRes.body.data.encoded_url);

    expect(decodeRes.body).toEqual({
        success: true,
        message: 'secret',
        qrExtracted: true,
        watermark: null // for now null if user didn't check watermark
      });
  });

  // test on Watermark validation
  it('Preserves watermark metadata', async () => {
    const encodeRes = await request(app)
      .post('/api/stegano/encode')
      .set('Authorization', `Bearer ${token}`)
      .field('message', 'secret')
      .field('watermark', 'user@email.com')
      .attach('image', testImageBuffer);

    const stegoImage = await StegoImage.findOne();
    expect(stegoImage.watermark.text).toBe('user@email.com');
    expect(stegoImage.userId.toString()).toBe((await User.findOne())._id.toString());
  });
});