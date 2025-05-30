import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:http_parser/http_parser.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:frontend/pages/encode_result.dart';

class HideMessageScreen extends StatefulWidget {
  @override
  _HideMessageScreenState createState() => _HideMessageScreenState();
}

class _HideMessageScreenState extends State<HideMessageScreen> {
  final TextEditingController _messageController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  File? _selectedImage;
  Map<String, dynamic>? _imageAnalysisResult;
  bool _generateQR = false;
  bool _addWatermarks = false;
  bool _isAnalyzing = false;


  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      setState(() {
        _selectedImage = File(picked.path);
        _isAnalyzing = true;
        _imageAnalysisResult = null;
      });
      await _analyzeImage(File(picked.path));
    }
  }

  Future<void> _analyzeImage(File image) async {
    // TODO when backend finish: Call backend API
    await Future.delayed(Duration(seconds: 2)); // mock delay
    setState(() {
      _imageAnalysisResult = {
        'capacity': '1024 bits',
        'resolution': '1080x720',
        'format': 'PNG',
      };
      _isAnalyzing = false;
    });
  }

  bool get _canEncode =>
      _selectedImage != null &&
          _imageAnalysisResult != null &&
          _messageController.text.isNotEmpty &&
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
            SizedBox(height: 20),
            Text(
              'Hide Your Message',
              style: TextStyle(
                  fontFamily: 'Orbitron',
                  fontWeight: FontWeight.w600,
                  color: Color(0xFFF4F4F4),
                  fontSize: 34,
              )
            ),
            SizedBox(height: 25),
            Text(
                'Secret Message',
                style: TextStyle(
                  fontFamily: 'Inter',
                  fontWeight: FontWeight.w400,
                  color: Color(0xFFF4F4F4),
                  fontSize: 18,
                )
            ),
            SizedBox(height: 10),
            _buildTextField(_messageController, 'Secret message'),
            SizedBox(height: 12),
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
            if (_isAnalyzing)
              Padding(
                padding: EdgeInsets.all(12),
                child: Center(child: CircularProgressIndicator()),
              ),
            if (_selectedImage != null && !_isAnalyzing) ...[
            SizedBox(height: 12),
            Image.file(_selectedImage!, height: 150),
            SizedBox(height: 8),
            if (_imageAnalysisResult != null)
            _buildAnalysisDetails(_imageAnalysisResult!),

            ],

            CheckboxListTile(
              value: _addWatermarks,
              onChanged: (val) => setState(() => _addWatermarks = val ?? false),
              title: Text('Add watermarks',
                  style: TextStyle(
                      color: Color(0xFFF4F4F4),
                  )
              ),
            ),
            SizedBox(height: 10),
            ElevatedButton(
              onPressed: _canEncode ? _sendEncodeRequest : null,
              style: ElevatedButton.styleFrom(
                backgroundColor:
                _canEncode ? Color(0xFF0CCE6B) : Color(0xFFB4B4B4),
                foregroundColor: _canEncode ? Color(0xFFF4F4F4) : Colors.grey[800],
                minimumSize: Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(4),
                ),
                splashFactory: _canEncode ? InkRipple.splashFactory : NoSplash.splashFactory,
              ),
              child: Text(
                  'Hide message',
                  style: TextStyle(
                    fontFamily: 'Orbitron',
                    fontWeight: FontWeight.w500,
                    fontSize: 18,
                  ),
              ),
            )
          ]
        )
      )
    );
  }
  Future<void> _sendEncodeRequest() async {
    if (_selectedImage == null) return;
    final prefs = await SharedPreferences.getInstance();
    final token = prefs.getString('token');

    if (token == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: Text('Authentication token not found. Please log in again.'),
      ));
      return;
    }

    final url = Uri.parse('http://10.0.2.2:5000/api/encode');

    final request = http.MultipartRequest("POST", url)
      ..headers['Authorization'] = 'Bearer $token'
      ..fields['message'] = _messageController.text
      ..fields['password'] = _passwordController.text
      ..fields['addQRCode'] = _generateQR.toString()
      ..fields['addWatermark'] = _addWatermarks.toString()
      ..files.add(await http.MultipartFile.fromPath(
        'image',
        _selectedImage!.path,
        contentType: MediaType('image', 'png'),
      ));

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);

        final String dataUrl = responseData['data']['base64'];
        final String base64Str = dataUrl.split(',')[1];
        final String downloadUrl = responseData['data']['downloadUrl'];

        if (!mounted) return;
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) => EncodeResultScreen(
              originalImage: _selectedImage!,
              base64EncodedImage: base64Str,
              downloadUrl: downloadUrl,
            ),
          ),
        );
      } else {
        throw Exception("Failed to encode message");
      }
    } catch (e) {
      print("Encoding error: $e");
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: Text("Error"),
          content: Text("An error occurred while encoding the message."),
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

  Widget _buildAnalysisDetails(Map<String, dynamic> data) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: data.entries.map((entry) {
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 2.0),
          child: Text("${entry.key}: ${entry.value}",
              style: TextStyle(color: Colors.white70)),
        );
      }).toList(),
    );
  }
}