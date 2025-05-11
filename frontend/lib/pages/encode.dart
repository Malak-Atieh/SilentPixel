import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';



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
    );
  }
}