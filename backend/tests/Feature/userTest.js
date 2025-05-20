const request = require('supertest');
const path = require('path');
const app = require('../../app');
const mongoose = require('mongoose');
const User = require('../../app/models/User');

describe('User Feature Tests', () => {
  beforeAll(async () => {
    await mongoose.connect(process.env.MONGO_URL);
  });

  afterAll(async () => {
    await mongoose.connection.close();
  });

  afterEach(async () => {
    await User.deleteMany();
  });

  describe('Registration', () => {
    it('successfully registers with valid credentials', async () => {
      const res = await request(app)
        .post('/api/register')
        .send({
          username: 'validuser',
          email: 'valid@example.com',
          password: 'SecurePass123!'
        });
      
      expect(res.statusCode).toBe(201);
      expect(res.body).toEqual({
        success: true,
        user: expect.objectContaining({
          username: 'validuser',
          email: 'valid@example.com'
        }),
        token: expect.any(String)
      });
    });

    it('rejects duplicate email registration', async () => {
      await User.create({
        username: 'existinguser',
        email: 'exists@example.com',
        password: await User.hashPassword('password123')
      });

      const res = await request(app)
        .post('/api/register')
        .send({
          username: 'newuser',
          email: 'exists@example.com',
          password: 'newpassword123'
        });

      expect(res.statusCode).toBe(400);
      expect(res.body.error).toMatch(/email already exists/i);
    });

    it('rejects weak passwords', async () => {
      const testCases = [
        { password: '123', reason: 'too short' },
        { password: 'password', reason: 'no numbers/special chars' },
        { password: 'abcdefgh', reason: 'no complexity' }
      ];

      for (const test of testCases) {
        const res = await request(app)
          .post('/api/register')
          .send({
            username: `user_${test.reason}`,
            email: `${test.reason.replace(/\s+/g, '')}@test.com`,
            password: test.password
          });
        
        expect(res.statusCode).toBe(400);
        expect(res.body.error).toMatch(/password/i);
      }
    });

    it('validates Base64 profile pictures', async () => {
      const testCases = [
        { 
          input: 'data:image/png;base64,invalidbase64', 
          expected: 'invalid Base64' 
        },
        { 
          input: 'not-a-base64-string', 
          expected: 'invalid format' 
        },
        { 
          input: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 
          expected: 'success' 
        }
      ];

      for (const test of testCases) {
        const res = await request(app)
          .post('/api/register')
          .send({
            username: `user_${test.expected}`,
            email: `${test.expected.replace(/\s+/g, '')}@test.com`,
            password: 'ValidPass123!',
            profilePicture: test.input
          });

        if (test.expected === 'success') {
          expect(res.statusCode).toBe(201);
        } else {
          expect(res.statusCode).toBe(400);
          expect(res.body.error).toMatch(new RegExp(test.expected, 'i'));
        }
      }
    });
  });

  describe('Authentication', () => {
    const testUser = {
      username: 'authuser',
      email: 'auth@test.com',
      password: 'AuthPass123!'
    };

    beforeEach(async () => {
      await request(app)
        .post('/api/register')
        .send(testUser);
    });

    it('successfully logs in with correct credentials', async () => {
      const res = await request(app)
        .post('/api/login')
        .send({
          email: testUser.email,
          password: testUser.password
        });

      expect(res.statusCode).toBe(200);
      expect(res.body).toEqual({
        success: true,
        token: expect.any(String),
        user: expect.objectContaining({
          email: testUser.email
        })
      });
    });

    it('rejects invalid passwords', async () => {
      const res = await request(app)
        .post('/api/login')
        .send({
          email: testUser.email,
          password: 'wrongpassword'
        });

      expect(res.statusCode).toBe(401);
      expect(res.body.error).toMatch(/invalid credentials/i);
    });

    it('rejects non-existent users', async () => {
      const res = await request(app)
        .post('/api/login')
        .send({
          email: 'nonexistent@test.com',
          password: 'anypassword'
        });

      expect(res.statusCode).toBe(401);
      expect(res.body.error).toMatch(/invalid credentials/i);
    });
  });
});