const User = require('../Models/User');
const { hashPassword, comparePassword } = require('../utils/hash');

async function registerUser(data) {
  data.password = await hashPassword(data.password);
  return await User.create(data);
}

async function authenticateUser(email, password) {
  const user = await User.findOne({ email });
  if (!user) throw new Error('Invalid credentials');
  
  const isMatch = await comparePassword(password, user.password);
  if (!isMatch) throw new Error('Invalid credentials');

  return user;
}

module.exports = { registerUser, authenticateUser };
