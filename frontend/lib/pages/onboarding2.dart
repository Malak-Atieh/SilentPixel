import 'package:flutter/material.dart';
import 'package:frontend/pages/onboarding3.dart';
import 'package:frontend/pages/sign_up.dart';

class OnboardingScreenSecond extends StatelessWidget {
  const OnboardingScreenSecond({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background image
          Positioned.fill(
            child: Image.asset(
              'assets/images/onboarding2.png',
              fit: BoxFit.cover,
            ),
          ),

          // Gradient overlay for better readability
          Positioned.fill(
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.black.withAlpha((0.2 * 255).toInt()), Colors.black.withAlpha((0.7 * 255).toInt())],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),
          ),

          // Content
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 40.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Spacer(),
                const Spacer(),
                const Spacer(),
                const Spacer(),
                const Spacer(),
                const Spacer(),
                // Title
                const Text(
                  'Embed messages\ninside your images',
                  style: TextStyle(
                    fontSize: 36,
                    fontWeight: FontWeight.w600,
                    color: Color(0xFFF4F4F4),
                    fontFamily: 'Orbitron',

                  ),
                ),
                const SizedBox(height: 16),

                // Subtitle
                const Text(
                  'Easily hide or reveal messages with just a few taps. Fast previews, powerful analysis, and complete privacy, all in one app.',
                  style: TextStyle(
                    fontFamily: 'Inter',
                    fontWeight: FontWeight.w500,
                    fontSize: 18,
                    color: Color(0xFFF4F4F4),
                  ),
                ),

                const Spacer(),

                // Bottom Navigation
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => SignUp()),
                        );
                      },
                      child: const Text(
                        'Skip',
                        style: TextStyle(
                          fontFamily: 'Inter',
                          fontWeight: FontWeight.w500,
                          fontSize: 16,
                          color: Color(0xFFF4F4F4)
                        ),
                      ),
                    ),
                    Row(
                      children: [
                        Container(
                          height: 8,
                          width: 8,
                          margin: const EdgeInsets.symmetric(horizontal: 4),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(10),
                            color: Color(0xFFF4F4F4),
                          ),
                        ),
                        Container(
                          height: 8,
                          width: 18,
                          margin: const EdgeInsets.symmetric(horizontal: 4),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(10),
                            color: Color(0xFFF4F4F4),
                          ),
                        ),
                        Container(
                          height: 8,
                          width: 8,
                          margin: const EdgeInsets.symmetric(horizontal: 4),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Color(0xFFF4F4F4),
                          ),
                        ),
                      ],
                    ),
                    GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(builder: (context) => OnboardingScreenThird()),
                        );
                      },
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Color(0xFFF4F4F4),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.arrow_forward_ios_rounded, size: 16),
                      ),
                    ),
                  ],
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
