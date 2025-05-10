import 'package:flutter/material.dart';

class Home extends StatelessWidget {
  const Home({super.key});
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
                color: Colors.white
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
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(
            children: [
            const SizedBox(height: 80),
            Center(
              child: Image.asset(
                'assets/images/whitelogo.png',
                width: 250,
                height: 200,
                fit: BoxFit.fill,
              ),
            ),
            const SizedBox(height: 20),
            const Text(
                'Silent Pixel',
                style: TextStyle(
                  fontFamily: 'Orbitron',
                  fontWeight: FontWeight.w400,
                  fontSize: 46,
                  color:Color(0xFFF4F4F4),
                ),
            ),
          ],
        ),
      ),
    );
  }
}