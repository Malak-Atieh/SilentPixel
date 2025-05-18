import 'dart:convert';
import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';

class EncodeResultScreen extends StatelessWidget {
  final File originalImage;
  final String base64EncodedImage;
  final String downloadUrl;

  const EncodeResultScreen({
    required this.originalImage,
    required this.base64EncodedImage,
    required this.downloadUrl,
  });

  @override
  Widget build(BuildContext context) {
    Uint8List encodedBytes = base64Decode(base64EncodedImage);

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
            Text("Hide Your Message",
              style: TextStyle(
              fontFamily: 'Orbitron',
              fontWeight: FontWeight.w600,
              color: Color(0xFFF4F4F4),
              fontSize: 34,
              ),
            ),
            SizedBox(height: 25),
            _buildImageBlock("Original Image", Image.file(originalImage)),
            SizedBox(height: 20),
            _buildImageBlock(
                "Encoded Image",
                Image.memory(encodedBytes, fit: BoxFit.cover)
            ),
            SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                _downloadImage(context, base64EncodedImage);
              },
              icon: Icon(
                Icons.download,
                color: Color(0xFFF4F4F4),
                size: 22,
              ),
              label: Text("Download",
                  style: TextStyle(
                    fontFamily: 'Orbitron',
                    fontWeight: FontWeight.w500,
                    fontSize: 18,
                    color: Color(0xFFF4F4F4),
                  )
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFF0CCE6B),
                padding: EdgeInsets.symmetric(vertical: 14, horizontal: 24),
                minimumSize: Size(double.infinity, 50),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
            ),
            SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _socialIcon(FontAwesomeIcons.instagram),
                _socialIcon(FontAwesomeIcons.facebook),
                _socialIcon(FontAwesomeIcons.whatsapp),
                _socialIcon(Icons.share),
              ],
            )
          ],
        ),
      ),
    );
  }
  // Helper function to detect if running on an emulator
  Future<bool> _isEmulator() async {
    try {
      final directory = await getExternalStorageDirectory();
      // Most emulators have specific paths that we can detect
      return directory?.path.contains('emulated') == true;
    } catch (e) {
      return false;
    }
  }

  // Helper function to detect Android version
  Future<bool> _isAndroid11OrHigher() async {
    // Android 11 is API level 30
    return await Permission.manageExternalStorage.status.isGranted;
  }

  // Save image without strict permission checks (for emulators)
  Future<void> _saveImageWithoutPermissionCheck(BuildContext context, Uint8List bytes) async {
    try {
      Directory? saveDir;

      // Try app's documents directory first
      saveDir = await getApplicationDocumentsDirectory();

      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final downloadPath = '${saveDir.path}/SilentPixel_encoded_$timestamp.png';

      final file = File(downloadPath);
      await file.writeAsBytes(bytes);

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text("Image saved to app directory: ${file.path}"),
          backgroundColor: Color(0xFF0CCE6B),
          duration: Duration(seconds: 5),
        ),
      );
    } catch (e) {
      throw e; // Rethrow to be caught by the outer try/catch
    }
  }
  Future<void> _downloadImage(BuildContext context, String base64Image) async {
      try {
        final bytes = base64Decode(base64Image);

        // Check if running on emulator (a simple detection method)
        bool isEmulator = await _isEmulator();

        // Simplified approach for emulators
        if (isEmulator) {
          // On emulators, we'll try to save directly without checking permissions
          // as the emulator permissions system can be unreliable
          await _saveImageWithoutPermissionCheck(context, bytes);
          return;
        }

        // For real devices: request all potentially needed permissions
        Map<Permission, PermissionStatus> statuses = await [
          Permission.storage,
          Permission.photos,
        ].request();

        // Also request manage external storage for Android 11+
        if (await _isAndroid11OrHigher()) {
          await Permission.manageExternalStorage.request();
        }

        // Check if basic permissions are granted (we'll try even if some fail)
        bool canProceed = statuses[Permission.storage]?.isGranted == true ||
            statuses[Permission.photos]?.isGranted == true;

        if (!canProceed) {
          // Show a dialog explaining the need for permissions
          await showDialog(
            context: context,
            builder: (BuildContext context) => AlertDialog(
              title: Text("Permission Required"),
              content: Text("This app needs storage permissions to save images. Please grant the permissions in the app settings and try again."),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text("Cancel"),
                ),
                TextButton(
                  onPressed: () async {
                    Navigator.pop(context);
                    await openAppSettings();
                  },
                  child: Text("Open Settings"),
                ),
              ],
            ),
          );
          return;
        }

        // Determine the best directory to save the file
        Directory? saveDir;
        try {
          // Try multiple locations in order of preference
          saveDir = await getExternalStorageDirectory();

          if (saveDir == null) {
            // Fallback to pictures directory
            saveDir = Directory('/storage/emulated/0/Pictures');
            if (!await saveDir.exists()) {
              await saveDir.create(recursive: true);
            }
          }
        } catch (e) {
          // Final fallback to app documents directory
          saveDir = await getApplicationDocumentsDirectory();
        }

        if (saveDir == null) {
          throw Exception("Could not find a suitable directory to save the image");
        }

        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final downloadPath = '${saveDir.path}/SilentPixel_encoded_$timestamp.png';

        final file = File(downloadPath);
        await file.writeAsBytes(bytes);

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Image saved successfully to ${file.path}"),
            backgroundColor: Color(0xFF0CCE6B),
            duration: Duration(seconds: 5),
          ),
        );
      } catch (e) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Failed to save image: ${e.toString()}"),
            backgroundColor: Colors.red,
            duration: Duration(seconds: 5),
          ),
        );
      }
  }

  Widget _buildImageBlock(String title, Widget imageWidget) {
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
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: SizedBox(
              height: 200,
              width: double.infinity,
              child: FittedBox(
                fit: BoxFit.contain,
                child: imageWidget,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _socialIcon(IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8.0),
      child: CircleAvatar(
        radius: 22,
        backgroundColor: Color(0xFF23488A),
        child: Icon(icon, color: Colors.white),
      ),
    );
  }
}
