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

                      TextField(
                        style: const TextStyle(
                          fontFamily: 'Inter',
                          fontWeight: FontWeight.w400,
                          color: Color(0xFFB4B4B4),
                        ),
                        decoration: InputDecoration(
                          hintText: 'JohnDoe',
                          hintStyle: const TextStyle(
                            color: Color(0xFFB4B4B4)
                            ),
                          labelText: 'Username',
                          labelStyle: const TextStyle(
                            color: Color(0xFFB4B4B4)
                            ),
                          filled: true,
                          fillColor: Color(0xFFF4F4F4),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                      ),

                      const SizedBox(height: 20),

                      TextField(
                        style: const TextStyle(
                          fontFamily: 'Inter',
                          fontWeight: FontWeight.w400,
                          color: Color(0xFFB4B4B4),
                        ),
                        decoration: InputDecoration(
                          hintText: 'JohnDoe@example.com',
                          hintStyle: const TextStyle(
                            color: Color(0xFFB4B4B4)
                            ),
                          labelText: 'Email',
                          labelStyle: const TextStyle(
                            color: Color(0xFFB4B4B4)
                            ),
                          filled: true,
                          fillColor: Color(0xFFF4F4F4),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                      ),

                      const SizedBox(height: 20),

                      TextField(
                        obscureText: true,
                        style: const TextStyle(
                          fontFamily: 'Inter',
                          fontWeight: FontWeight.w400,
                          color:  Color(0xFFB4B4B4),
                        ),
                        decoration: InputDecoration(
                          hintText: 'enter 8 digit password',
                          hintStyle: const TextStyle(
                            color: Color(0xFFB4B4B4)
                            ),
                          labelText: 'Password',
                          labelStyle: const TextStyle(
                            color: Color(0xFFB4B4B4)
                            ),
                          filled: true,
                          fillColor: Color(0xFFF4F4F4),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                      ),

                      const SizedBox(height: 10),

                      ElevatedButton(
                        onPressed: () {},
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Color(0xFF0CCE6B),
                          minimumSize: const Size(double.infinity, 50),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10)
                          ),
                        ),
                        child: const Text(
                          'SignUp',
                          style: TextStyle(
                            fontFamily: 'Orbitron',
                            color: Color(0xFFF4F4F4),
                            fontSize: 18
                          ),
                          
                        ),
                      ),

                      const SizedBox(height: 20),

                      // Or login with divider
                      Row(
                        children: const [
                          Expanded(
                              child: Divider(
                                  color: Color(0xFFF4F4F4)
                              )
                          ),
                          Padding(
                            padding: EdgeInsets.symmetric(horizontal: 10),
                            child: Text(
                                "Or Sign Up with",
                                style: TextStyle(
                                    color: Color(0xFFF4F4F4)
                                )
                            ),
                          ),
                          Expanded(
                              child: Divider(
                                  color: Color(0xFFF4F4F4)
                              )
                          ),
                        ],
                      ),

                      const SizedBox(height: 20),

                      // Social icons
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceAround,
                        children: [
                          _buildSocialIcon(FontAwesomeIcons.google),
                          _buildSocialIcon(FontAwesomeIcons.instagram),
                          _buildSocialIcon(FontAwesomeIcons.facebookF),
                        ],
                      ),

                      const Spacer(),


                      // Sign up prompt
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Text(
                              "Don't have an account? ",
                              style: TextStyle(
                                  color: Color(0xFFF4F4F4)
                              )
                          ),
                          GestureDetector(
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => SignUp()),
                              );
                            },
                            child: const Text(
                                "Sign Up",
                                style: TextStyle(
                                    color: Color(0xFF0CCE6B)
                                )
                            ),
                          ),
                        ],
                      ),

                      const SizedBox(height: 20),
                    ],
                  ),
                ),

            ),



          ],
      ),
    );
  }
}