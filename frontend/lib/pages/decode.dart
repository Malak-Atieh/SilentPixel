import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:http_parser/http_parser.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:frontend/pages/decode_result.dart';

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
  bool get _canDecode =>
      _selectedImage != null &&
          _passwordController.text.length >= 8;


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
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: _canDecode ? _sendDecodeRequest : null,
              style: ElevatedButton.styleFrom(
                backgroundColor:
                _canDecode ? Color(0xFF0CCE6B) : Color(0xFFB4B4B4),
                foregroundColor: _canDecode ? Color(0xFFF4F4F4) : Colors.grey[800],
                minimumSize: Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(4),
                ),
                splashFactory: _canDecode ? InkRipple.splashFactory : NoSplash.splashFactory,
              ),
              child: Text(
                'Reveal message',
                style: TextStyle(
                  fontFamily: 'Orbitron',
                  fontWeight: FontWeight.w500,
                  fontSize: 18,
                ),
              ),
            ),
          ],
        )
      ),
    );
  }

  Future<void> _sendDecodeRequest() async {
    if (_selectedImage == null) return;

    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
    });

    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('token');

    if (token == null) {
      setState(() {
        _isAnalyzing = false;
      });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Authentication token not found. Please log in again.'),
        backgroundColor: Colors.red,
      ));
      return;
    }

    final url = Uri.parse('http://10.0.2.2:5000/api/decode');

    final request = http.MultipartRequest("POST", url)
      ..headers['Authorization'] = 'Bearer $token'
      ..fields['password'] = _passwordController.text
      ..files.add(await http.MultipartFile.fromPath(
        'image',
        _selectedImage!.path,
        contentType: MediaType('image', 'png'),
      ));

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      print("Response status: ${response.statusCode}");
      print("Response body: ${response.body}");

      setState(() {
        _isAnalyzing = false;
      });

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);

        if (!responseData.containsKey('data')) {
          throw Exception("Invalid response format: 'data' field missing");
        }
        final data = responseData['data'];
        if (!data.containsKey('message')) {
          throw Exception("Invalid response format: 'message' field missing");
        }

        final String message = data['message'];

        // Handle watermark - use an empty string if it's null or missing
        final String watermark = data.containsKey('watermark') && data['watermark'] != null
            ? data['watermark']
            : '';

        if (!mounted) return;
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => DecodeResultScreen(
              message: message,
              watermark: watermark,
            ),
          ),
        );
      } else {
        throw Exception("Failed to decode image");
      }
    } catch (e) {
      print("Decoding error: $e");
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: Text("Error"),
          content: Text("An error occurred while decoding the image."),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text("OK"),
            )
          ],
        ),
      );
    }
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