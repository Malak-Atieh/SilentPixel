import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:frontend/pages/sign_up.dart';
import 'package:frontend/pages/home.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class Login extends StatefulWidget {
  const Login({super.key});

  @override
  State<Login> createState() => _LoginState();
}
class _LoginState extends State<Login>  {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _rememberMe = false;
  bool get _canLogin =>
          _emailController.text.isNotEmpty &&
          _passwordController.text.length >= 8;
  @override
  void initState() {
    super.initState();
    _loadRememberMe();
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: Stack(
          children: [
            // Background image
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
            // Login form
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Spacer(),
                  
                  const SizedBox(height: 40),
                  const Center(
                    child: Text(
                      'Login',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 36,
                        fontWeight: FontWeight.w600,
                        color: Color(0xFFF4F4F4),
                        fontFamily: 'Orbitron',
                      ),
                    ),
                  ),
                  const SizedBox(height: 40),
                  Text(
                      'Email',
                      textAlign: TextAlign.left,
                      style: TextStyle(
                        fontFamily: 'Inter',
                        fontWeight: FontWeight.w400,
                        color: Color(0xFFF4F4F4),
                        fontSize: 18,
                      )
                  ),
                  SizedBox(height: 7),
                  _buildTextField(_emailController, 'Email', hint: 'John@example.com'),
                  const SizedBox(height: 10),
                  Text(
                      'Password',
                      textAlign: TextAlign.left,
                      style: TextStyle(
                        fontFamily: 'Inter',
                        fontWeight: FontWeight.w400,
                        color: Color(0xFFF4F4F4),
                        fontSize: 18,
                      )
                  ),
                  SizedBox(height: 7),
                  _buildTextField(_passwordController, 'Password',
                      hint: 'Enter 8 digit password', obscure: true),
                  const SizedBox(height: 10),

                  // Remember and forgot password
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Transform.scale(
                            scale: 0.8,
                            child: Switch(
                              value: _rememberMe,
                              onChanged: (val) {
                                setState(() {
                                  _rememberMe = val;
                                });
                              },
                              activeColor: Colors.grey[400],
                              activeTrackColor: Color(0xFF23488A),
                            ),
                          ),
                          const Text('Remember me', style: 
                            TextStyle(
                              fontFamily: 'Inter',
                              fontWeight: FontWeight.w400,
                              color: Color(0xFFF4F4F4),
                              fontSize: 14,
                            )
                          ),
                        ],
                      ),
                      TextButton(
                        onPressed: () {},
                        child: const Text(
                          'Forget password?',
                          style: TextStyle(
                            fontFamily: 'Inter',
                            fontWeight: FontWeight.w400,
                            color: Color(0xFFF4F4F4),
                            fontSize: 14,
                          ),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 10),

                  // Login button
                  ElevatedButton(
                    onPressed: () {
                      if (_canLogin) {
                        _submitLogin();
                      } else {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text("Please fill all fields correctly!"),
                            backgroundColor: Colors.red,
                          ),
                        );
                      }
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _canLogin ? Color(0xFF0CCE6B) : Color(0xFF8C8C8C),
                      minimumSize: const Size(double.infinity, 50),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    child: const Text(
                      'Login', 
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
                          "Or login with", 
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

  Future<void> _loadRememberMe() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _rememberMe = prefs.getBool('rememberMe') ?? false;
      if (_rememberMe) {
        _emailController.text = prefs.getString('savedEmail') ?? '';
        _passwordController.text = prefs.getString('savedPassword') ?? '';
      }
    });
  }

  Future<void> _saveRememberMe() async {
    final prefs = await SharedPreferences.getInstance();
    if (_rememberMe) {
      await prefs.setString('savedEmail', _emailController.text);
      await prefs.setString('savedPassword', _passwordController.text);
    } else {
      await prefs.remove('savedEmail');
      await prefs.remove('savedPassword');
    }
    await prefs.setBool('rememberMe', _rememberMe);
  }

  Future<void> _submitLogin() async {
    final url = Uri.parse('http://10.0.2.2:5000/api/login');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': _emailController.text,
          'password': _passwordController.text,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final token = data['data']['token'];
        final user = data['data']['user'];

        final prefs = await SharedPreferences.getInstance();
        await prefs.setString('token', token);
        await prefs.setString('user', json.encode(user));

        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Login successful!')),
        );

        Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => const Home())
        );

      } else {
        final data = json.decode(response.body);
        if (!mounted) return;
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Login Failed'),
            content: Text(data['message'] ?? 'Unknown error'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
    } catch (e) {
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Network Error'),
          content: Text('Could not connect to server: $e'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      );
    }
  }

  Widget _buildTextField(TextEditingController controller, String label,
      {required String hint, bool obscure = false}) {
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
      ),
      onChanged: (_) => setState(() {}),
    );
  }
}