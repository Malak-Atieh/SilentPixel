import 'package:flutter/material.dart';

class DecodeResultScreen extends StatelessWidget {
  final String message;
  final String watermark;

  const DecodeResultScreen({
    required this.message,
    required this.watermark,
  });

  @override
  Widget build(BuildContext context) {
    final String displayWatermark = (watermark.trim().isEmpty)
        ? "No watermark detected"
        : watermark;
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
        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 20),
        child: Column(
          children: [
            Text("Reveal Message",
              style: TextStyle(
                fontFamily: 'Orbitron',
                fontWeight: FontWeight.w600,
                color: Color(0xFFF4F4F4),
                fontSize: 34,
              ),
            ),
            SizedBox(height: 20),
            _buildBlock("Hidden message:", message),
            SizedBox(height: 20),
            _buildBlock(
                "Watermark detected:", displayWatermark),
            SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
  Widget _buildBlock(String title, content) {

    return Container(
      width: double.infinity,
      padding: EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Color(0xFFF4F4F4),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Column(
        children: [
          Text(title,
              style: TextStyle(
                  fontWeight: FontWeight.w700,
                  fontFamily: 'Orbitron',
                  fontSize: 15)
          ),
          SizedBox(height: 12),
          Text(content,
              style: TextStyle(
                  fontWeight: FontWeight.w700,
                  fontFamily: 'Orbitron',
                  fontSize: 15)
          ),
          SizedBox(height: 12),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: SizedBox(
              width: double.infinity,
              child: FittedBox(
                fit: BoxFit.contain,
              ),
            ),
          ),
        ],
      ),
    );
  }


}
