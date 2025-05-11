import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class RevealMessageScreen extends StatefulWidget {
  const RevealMessageScreen({super.key});

  @override
  State<RevealMessageScreen> createState() => _RevealMessageScreenState();
}

class _RevealMessageScreenState extends State<RevealMessageScreen> {
  final TextEditingController _passwordController = TextEditingController();
  File? _selectedImage;
  bool _isAnalyzing = false;
  bool _hasResult = false;
  String? _extractedMessage;
  String? _errorMessage;

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _isAnalyzing = true;
        _hasResult = false;
        _extractedMessage = null;
        _errorMessage = null;
      });


      await Future.delayed(const Duration(seconds: 2)); // mock api call

      setState(() {
        _isAnalyzing = false;
        _hasResult = true;
        _extractedMessage = 'Hello, this is your hidden message!'; // mock response
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF09192C),
      appBar: AppBar(
        backgroundColor: const Color(0xFF09192C),
        elevation: 0,
        title: const Text(
            "Silent Pixel",
            style: TextStyle(
                fontFamily: 'Orbitron',
                fontWeight: FontWeight.w400,
                fontSize: 16,
                color: Color(0xFFF4F4F4)
            )
        ),
        leading: IconButton(
          icon: const Icon(Icons.menu, color:Color(0xFFF4F4F4)),
          onPressed: () {},
        ),
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12.0),
            child: CircleAvatar(
              backgroundColor: Colors.white24,
              child: Icon(Icons.person, color:Color(0xFFF4F4F4)), //later: radius: 20,backgroundImage: NetworkImage(user.profileImageUrl),
            ),
          )
        ],
      ),


      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(height: 35),
            Text(
              'Reveal Message',
              style: TextStyle(
                fontFamily: 'Orbitron',
                fontWeight: FontWeight.w600,
                color: Color(0xFFF4F4F4),
                fontSize: 34,
              )
            ),
            SizedBox(height: 30),
            Text(
                'Password',
                style: TextStyle(
                  fontFamily: 'Inter',
                  fontWeight: FontWeight.w400,
                  color: Color(0xFFF4F4F4),
                  fontSize: 18,
                )
            ),
            SizedBox(height: 10),
            _buildTextField(_passwordController, 'Enter your password',
                obscure: true),
            SizedBox(height: 25),
            ElevatedButton.icon(
              onPressed: _pickImage,
              icon: Icon(
                Icons.upload,
                color: Color(0xFFF4F4F4),
                size: 24,
              ),
              label: Text(
                'Upload image',
                style: TextStyle(
                  fontFamily: 'Orbitron',
                  fontWeight: FontWeight.w600,
                  fontSize: 18,
                  color: Color(0xFFF4F4F4),
                ),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFF23488A),
                minimumSize: Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(4), // 4px radius
                ),
              ),
            ),
            const SizedBox(height: 20),
            if (_selectedImage != null)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Image.file(_selectedImage!, height: 180),
                  const SizedBox(height: 10),
                  if (_isAnalyzing)
                    const Center(child: CircularProgressIndicator())
                  else if (_hasResult && _extractedMessage != null)
                    Text('Extracted Message: $_extractedMessage', style: const TextStyle(color: Colors.white))
                  else if (_errorMessage != null)
                      Text(_errorMessage!, style: const TextStyle(color: Colors.red)),
                ],
              ),
          ],
        )
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String hint,
      {bool obscure = false}) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      style: TextStyle(
        color: Color(0xFF09192C),
      ),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(
          color: Color(0xFFB4B4B4),
        ),
        filled: true,
        fillColor: Color(0xFFF4F4F4),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(4)),
      ),
    );
  }

}