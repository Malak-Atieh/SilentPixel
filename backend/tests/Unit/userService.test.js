// ✅ Always mock before importing the module that uses the dependency
jest.mock('../../app/utils/hash', () => ({
  hashPassword: jest.fn(),
  comparePassword: jest.fn(),
}));

jest.mock('../../app/models/User'); // You had Models capitalized incorrectly in some places

const UserService = require('../../app/services/UserService');
const { ValidationError, AuthError } = require('../../app/Traits/errors');
const User = require('../../app/models/User');
const { hashPassword, comparePassword } = require('../../app/utils/hash'); // ✅ Import mocked functions AFTER jest.mock

describe('UserService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('registerUser', () => {
    it('should register a user successfully', async () => {
      const userData = {
        email: 'test@example.com',
        username: 'testuser',
        password: 'Password123'
      };

      const hashedPassword = 'hashed_password';
      const mockUser = {
        _id: 'user123',
        email: userData.email,
        username: userData.username,
        password: hashedPassword,
        toObject: function () {
          return {
            _id: this._id,
            email: this.email,
            username: this.username,
            password: this.password
          };
        }
      };

      // ✅ Setup mocks
      User.findOne.mockResolvedValue(null);
      hashPassword.mockResolvedValue(hashedPassword);
      User.create.mockResolvedValue(mockUser);

      const result = await UserService.registerUser(userData);

      expect(User.findOne).toHaveBeenCalledWith({
        $or: [
          { email: userData.email },
          { username: userData.username }
        ]
      });
      expect(hashPassword).toHaveBeenCalledWith(userData.password);
      expect(User.create).toHaveBeenCalledWith(expect.objectContaining({
        email: userData.email,
        username: userData.username,
        password: hashedPassword,
      }));
      expect(result).toEqual({
        _id: 'user123',
        email: userData.email,
        username: userData.username
      });
      expect(result.password).toBeUndefined();
    });

    it('should throw ValidationError if required fields are missing', async () => {
      const incompleteUserData = {
        email: 'test@example.com',
        password: 'Password123'
      };

      await expect(UserService.registerUser(incompleteUserData))
        .rejects
        .toThrow(ValidationError);
    });

    it('should throw ValidationError if user already exists', async () => {
      const userData = {
        email: 'existing@example.com',
        username: 'existinguser',
        password: 'Password123'
      };

      User.findOne.mockResolvedValue({
        _id: 'existing123',
        email: userData.email
      });

      await expect(UserService.registerUser(userData))
        .rejects
        .toThrow(ValidationError);
    });
  });

  // 🟨 You don’t need to change the other blocks unless you face issues,
  // but ensure comparePassword is used from the mocked import

  describe('authenticateUser', () => {
    it('should authenticate user with valid credentials', async () => {
      const email = 'test@example.com';
      const password = 'Password123';

      const mockUser = {
        _id: 'user123',
        email,
        username: 'testuser',
        password: 'hashed_password',
        toObject: function () {
          return {
            _id: this._id,
            email: this.email,
            username: this.username,
            password: this.password
          };
        }
      };

      User.findOne.mockResolvedValue(mockUser);
      comparePassword.mockResolvedValue(true);

      const result = await UserService.authenticateUser(email, password);

      expect(User.findOne).toHaveBeenCalledWith({ email });
      expect(comparePassword).toHaveBeenCalledWith(password, mockUser.password);
      expect(result).toEqual({
        _id: 'user123',
        email,
        username: 'testuser'
      });
      expect(result.password).toBeUndefined();
    });

    it('should throw ValidationError if email or password is missing', async () => {
      await expect(UserService.authenticateUser('', 'password'))
        .rejects
        .toThrow(ValidationError);

      await expect(UserService.authenticateUser('email@example.com', ''))
        .rejects
        .toThrow(ValidationError);
    });

    it('should throw AuthError if user not found', async () => {
      User.findOne.mockResolvedValue(null);

      await expect(UserService.authenticateUser('nonexistent@example.com', 'password'))
        .rejects
        .toThrow(AuthError);
    });

    it('should throw AuthError if password is incorrect', async () => {
      const mockUser = {
        _id: 'user123',
        email: 'test@example.com',
        password: 'hashed_password'
      };

      User.findOne.mockResolvedValue(mockUser);
      comparePassword.mockResolvedValue(false);

      await expect(UserService.authenticateUser('test@example.com', 'wrongpassword'))
        .rejects
        .toThrow(AuthError);
    });
  });

  describe('getUserById', () => {
    it('should return user without password when valid userId is provided', async () => {
      const userId = 'user123';
      const mockUser = {
        _id: userId,
        email: 'test@example.com',
        username: 'testuser',
        password: 'hashed_password',
        toObject: () => ({
          _id: userId,
          email: 'test@example.com',
          username: 'testuser',
          password: 'hashed_password'
        })
      };

      User.findById.mockResolvedValue(mockUser);

      const result = await UserService.getUserById(userId);

      expect(User.findById).toHaveBeenCalledWith(userId);
      expect(result).toEqual({
        _id: userId,
        email: 'test@example.com',
        username: 'testuser'
      });
      expect(result.password).toBeUndefined();
    });

    it('should throw ValidationError if user not found', async () => {
      User.findById.mockResolvedValue(null);

      await expect(UserService.getUserById('nonexistent'))
        .rejects
        .toThrow(ValidationError);
    });
  });
});
