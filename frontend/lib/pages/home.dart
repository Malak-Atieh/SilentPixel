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
                fontWeight: FontWeight.w500,
                fontSize: 46,
                color:Color(0xFFF4F4F4),
              ),
            ),
            const Text(
              'Every pixel tells your story... Silently',
              style: TextStyle(
                fontFamily: 'Orbitron',
                fontWeight: FontWeight.w400,
                fontSize: 18,
                color:Color(0xFFF4F4F4),
              ),
            ),
            const SizedBox(height: 40),
            ElevatedButton(
              onPressed: () {},
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xFF0CCE6B),
                  minimumSize: const Size(double.infinity, 50),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                ),
                child: const Text(
                    'Hide a message',
                    style: TextStyle(
                        fontFamily: 'Orbitron',
                        color: Color(0xFFF4F4F4),
                        fontSize: 18
                    )
                ),
              ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }
}