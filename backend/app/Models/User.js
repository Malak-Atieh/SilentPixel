const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  username: { 
    type: String, 
    unique: true, 
    required: true,
    trim: true
  },
  email: { 
    type: String, 
    unique: true, 
    required: true, 
    lowercase: true,
    match: [/^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,3})+$/, 'Invalid email']
  },
  password: { 
    type: String, 
    required: true 
  }, 
  profilePicture: { type: String } //base64 from front
},{ timestamps: true });

module.exports = mongoose.model("User", userSchema);
