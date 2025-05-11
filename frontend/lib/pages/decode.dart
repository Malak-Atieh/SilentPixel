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

  @override
  Widget build(BuildContext context) {
    return Scaffold(

    );
  }
}