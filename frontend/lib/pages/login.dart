import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class Login extends StatelessWidget {
  const Login({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      body: SafeArea(
         child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const SizedBox(height: 40),
              const Text(
                'Login',
                style: TextStyle(
                fontSize: 36,
                fontWeight: FontWeight.w600,
                color: Color(0xFFF4F4F4),
                fontFamily: 'Orbitron',
                ),
              ),
              const SizedBox(height: 40),
              TextField(
                style: TextStyle(
                  fontFamily: 'Inter',
                  fontWeight: FontWeight.w400,
                  color: Color(0xFFF4F4F4)
                  ),
                decoration: InputDecoration(
                  hintText: 'JohnDoe@example.com',
                  hintStyle: TextStyle(color: Color(0xFFF4F4F4)),
                  labelText: 'Email',
                  labelStyle: TextStyle(color: Color(0xFFF4F4F4)),
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
                style: TextStyle(
                  fontFamily: 'Inter',
                  fontWeight: FontWeight.w400,
                  color: Color(0xFFF4F4F4)
                  ),
                decoration: InputDecoration(
                  hintText: 'enter 8 digit password',
                  hintStyle: TextStyle(color: Color(0xFFF4F4F4)),
                  labelText: 'Password',
                  labelStyle: TextStyle(color: Color(0xFFF4F4F4)),
                  filled: true,
                  fillColor: Color(0xFFF4F4F4),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Switch(value: false, onChanged: (_) {}),
                      const Text('Remember me', style: TextStyle(color: Color(0xFFF4F4F4))),
                    ],
                  ),
                  TextButton(
                    onPressed: () {},
                    child: const Text(
                      'Forget password?',
                      style: TextStyle(color: Color(0xFFF4F4F4)),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              ElevatedButton(
                onPressed: () {},
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  minimumSize: const Size(double.infinity, 50),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                ),
                child: const Text('Login', style: TextStyle(fontSize: 18)),
              ),
              const SizedBox(height: 20),
              Row(
                children: const [
                  Expanded(child: Divider(color: Color(0xFFF4F4F4))),
                  Padding(
                    padding: EdgeInsets.symmetric(horizontal: 10),
                    child: Text("Or login with", style: TextStyle(color: Color(0xFFF4F4F4))),
                  ),
                  Expanded(child: Divider(color: Color(0xFFF4F4F4))),
                ],
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildSocialIcon(FontAwesomeIcons.google),
                  _buildSocialIcon(FontAwesomeIcons.instagram),
                  _buildSocialIcon(FontAwesomeIcons.facebookF),
                ],
              ),
              const Spacer(),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text("Don't have an account? ", style: TextStyle(color: Color(0xFFF4F4F4))),
                  GestureDetector(
                    onTap: () {},
                    child: const Text("Sign Up", style: TextStyle(color: Colors.green)),
                  ),
                ],
              ),
              const SizedBox(height: 20),
            ]
          ),    
        ),     
      )
    );
  }

    Widget _buildSocialIcon(IconData icon) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.blueGrey[800],
        shape: BoxShape.circle,
      ),
      child: FaIcon(icon, color: Color(0xFFF4F4F4), size: 20),
    );
  }
}