import 'dart:async';

import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:frontend/pages/login.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class SignUp extends StatefulWidget {
  const SignUp({super.key});

  @override
  State<SignUp> createState() => _SignUpState();
}
class _SignUpState extends State<SignUp>  {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  bool get _isEmailValid =>
      RegExp(r"^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$")
          .hasMatch(_emailController.text.trim());
  bool _obscurePassword = true;
  bool get _canSignUp =>
      _usernameController.text.isNotEmpty &&
          _emailController.text.isNotEmpty &&
          _passwordController.text.length >= 8;
  @override
  void dispose() {
    _usernameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }
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
                      _buildLabel('Username'),
                      SizedBox(height: 10),
                      _buildTextField(_usernameController, 'Username', hint: 'JohnDoe'),

                      const SizedBox(height: 20),
                      _buildLabel('Email'),
                      SizedBox(height: 10),
                      _buildTextField(_emailController, 'Email', hint: 'John@example.com'),
                      const SizedBox(height: 20),
                      _buildLabel('Password'),
                      SizedBox(height: 10),
                      _buildTextField(_passwordController, 'Password',
                          hint: 'Enter 8 digit password',
                          obscure: _obscurePassword,
                          suffixIcon: IconButton(
                            icon: Icon(
                              _obscurePassword ? Icons.visibility_off : Icons.visibility,
                              color: const Color(0xFF09192C),
                            ),
                            onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                          )),
                      const SizedBox(height: 20),

                      ElevatedButton(
                        onPressed: _canSignUp ? () {
                          _submitSignup();
                        } : null,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: _canSignUp ? Color(0xFF0CCE6B) : Color(0xFFC8C8C8),
                          minimumSize: const Size(double.infinity, 50),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10)
                          ),
                        ),
                        child: const Text(
                          'Sign Up',
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
                              "Already have an account? ",
                              style: TextStyle(
                                  color: Color(0xFFF4F4F4)
                              )
                          ),
                          GestureDetector(
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => Login()),
                              );
                            },
                            child: const Text(
                                "Login",
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

  Widget _buildLabel(String label) {
    return Text(
      label,
      textAlign: TextAlign.left,
      style: const TextStyle(
        fontFamily: 'Inter',
        fontWeight: FontWeight.w400,
        color: Color(0xFFF4F4F4),
        fontSize: 18,
      ),
    );
  }

  Future<void> _submitSignup() async {
    try {
    final url = Uri.parse('http://10.0.2.2:5000/api/register');

    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'username': _usernameController.text.trim(),
        'email': _emailController.text.trim(),
        'password': _passwordController.text.trim(),
      }),
    ).timeout(const Duration(seconds: 10));
    final resData = jsonDecode(response.body);

    if (response.statusCode == 201) {
      // success
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
            content: Text(resData['message'] ?? 'Signed up successfully'),
            backgroundColor: Colors.green,
        ),

      );
      Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const Login())
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(resData['error'] ?? 'Signup failed'),
          backgroundColor: Colors.red,
        ),
      );
    }
    } on TimeoutException {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Request timed out')),
      );
    } on Exception catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    }
  }
  Widget _buildSocialIcon(IconData icon) {
    return Container(
      padding: const EdgeInsets.only(
          left: 45,
          right: 45,
          top:14,
          bottom: 14
      ),
      decoration: BoxDecoration(
        color: Color(0xFF23488A),
        borderRadius: BorderRadius.circular(10),
      ),
      child: FaIcon(
          icon,
          color: Color(0xFFF4F4F4),
          size: 20
      ),
    );
  }

  Widget _buildTextField(
      TextEditingController controller,
      String label, {
        required String hint,
        bool obscure = false,
        Widget? suffixIcon,
      }) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      style: TextStyle(
        color: Color(0xFF09192C),
        fontFamily: 'Inter',
        fontWeight: FontWeight.w400,
      ),

      decoration: InputDecoration(
        hintText: hint,
        hintStyle: TextStyle(
          color: Color(0xFFB4B4B4),
        ),
        filled: true,
        fillColor: Color(0xFFF4F4F4),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(4)),
        suffixIcon: suffixIcon,
      ),
      onChanged: (_) => setState(() {}),
    );
  }
}