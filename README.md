<img src="./readme/title1.svg"/>

<br><br>

<!-- project overview -->
<img src="./readme/title2.svg"/>

> System that allows users to hide and retrieve secret messages within images shared on popular social media platforms using steganography techniques. Users are also able to anaylze images for possible stego hidden content.

<br><br>

<!-- System Design -->
<img src="./readme/title3.svg"/>

### Database Schema

The database is structured around core collections:

- User: Manages authentication, profiles, and platform access with unique identifiers, credentials, and profile customization.
StegoImage: Serves as the central collection storing all steganographically processed images with comprehensive metadata including:

- Original image properties (filename, dimensions, format)
Steganography details (hidden content indicators, encryption status, modified regions)
Marking systems (watermarks, QR codes, positioning)
Processing information (methods, ML usage, processing times)
Lifecycle controls (public/private status, self-destruct mechanisms)



The schema implements relationship tracking between users and their created images, strategic indexing for performance optimization, and embedded documents for efficient retrieval of related data. This structure supports the platform's core functionality of secure message hiding, image processing, and content management while maintaining clear ownership and access controls.

<br><br>

<!-- Project Highlights -->
<img src="./readme/title4.svg"/>


<img src="./readme/Group 51.png"/>
### Make Pixel Talk

- Real-Time anaylsis to identify setganography presence
- Encrypt and decrypt messages in provided images
- Dynamic Watermarking with Hidden Messages

<br><br>

<!-- Demo -->
<img src="./readme/title5.svg"/>

### User Screens (Mobile)

| Login screen                            | Register screen                       | Home screen                           |
| --------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![Landing](./readme/Login.png)          | ![fsdaf](./readme/signUp.png)         | ![fsdaf](./readme/home.png)           |

| Hide message screen                     | Hide Result screen                    | Reveal message screen                 |
| --------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![Landing](./readme/hideMessage.png)    | ![fsdaf](./readme/EncodeResult.png)  | ![fsdaf](./readme/decode.png)         |

| Reveal Result screen                    | Anaylze screen                        | Home screen                           |
| --------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![Landing](./readme/decodeResult.png)   | ![fsdaf](./readme/demo/1440x1024.png)         | ![fsdaf](./readme/demo/1440x1024.png)           |

<br><br>

<!-- Development & Testing -->
<img src="./readme/title6.svg"/>

### Add Title Here


| Services                                | Validation                            | Testing                               |
| --------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![Landing](./readme/addQr.png)          | ![fsdaf](./readme/validation.png)     | ![fsdaf](./readme/userTest.png) |


<br><br>

<!-- Deployment -->
<img src="./readme/title7.svg"/>

### Add Title Here

- Description here.


| Postman API 1                           | Postman API 2                         | Postman API 3                        |
| --------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![Landing](./readme/loginPostman.png)   | ![fsdaf](./readme/encodePostman.png)  | ![fsdaf](./readme/decoded.png) |

<br><br>
