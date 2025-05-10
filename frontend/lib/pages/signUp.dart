import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:frontend/pages/login.dart';

class SignUp extends StatelessWidget {
  const SignUp({super.key});

    @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: Stack(
          children: [
            Positioned.fill(
              child: Image.asset(
                'assets/images/bg.png',
                fit: BoxFit.cover,
              ),
            ),
            Positioned.fill(
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Color(0xFF09192C).withAlpha((0.6 * 255).toInt()),
                      Color(0xFF09192C).withAlpha((0.9 * 255).toInt())
                    ],
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                  ),
                ),
              ),
            ),
            SafeArea(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      const Spacer(),
                      
                      const SizedBox(height: 40),
                      const Text(
                        'Sign Up',
                        style: TextStyle(
                          fontSize: 36,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFFF4F4F4),
                          fontFamily: 'Orbitron',
                        ),
                      ),
                      const SizedBox(height: 40),
                    ],
                  ),
                ),

            ),


          ],
      ),
    );
  }
}