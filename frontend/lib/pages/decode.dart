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
      ),
    );
  }
}