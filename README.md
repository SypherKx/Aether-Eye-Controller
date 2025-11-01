# Aether Eye Controller 👁️

A powerful AI-powered vision system combining YOLO object detection, ESP32 camera streaming, and a React Native mobile controller. Control your ESP32 camera, perform room scans with AI object detection, and get voice feedback - all from your mobile device.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### Mobile App
- **Real-time ESP32 Camera Control** - Connect and control your ESP32 camera module
- **Room Scanning** - AI-powered object detection using YOLO
- **Flashlight Control** - Auto/On/Off modes for optimal lighting
- **Motion Detection** - Enable/disable motion sensing
- **Voice Feedback** - Text-to-speech for scan results
- **WebSocket Integration** - Real-time updates and notifications
- **Modern UI** - Beautiful dark theme interface

### Server
- **FastAPI Backend** - High-performance REST API server
- **YOLO Object Detection** - Real-time object recognition
- **ESP32 Integration** - Seamless camera stream handling
- **WebSocket Support** - Live updates for mobile clients
- **Motion Detection** - Smart motion sensing capabilities

## 📋 Prerequisites

- **Python 3.8+** (for server)
- **Node.js 18+** and **npm** (for mobile app)
- **ESP32 Camera Module** (connected to local network)
- **Expo CLI** (for mobile development)
- **Android Studio** (optional, for local builds)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/SypherKx/Aether-Eye-Controller.git
cd Aether-Eye-Controller
```

### 2. Set Up Python Server

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure ESP32 Camera

Edit `hello.py` and set your ESP32 camera IP:

```python
CAM_SOURCE = os.getenv("ESP32_CAM_URL", "http://YOUR_ESP32_IP:81/stream")
```

### 4. Start the Server

```bash
python hello.py
```

Server will start at `http://0.0.0.0:8000`

### 5. Set Up Mobile App

```bash
cd mobile
npm install
```

### 6. Run Mobile App

**Development Mode:**
```bash
npm start
# Then press 'a' for Android or 'i' for iOS
```

**Build Android APK:**
```bash
# Using EAS (cloud build)
npm run build:android

# Or locally (requires Android Studio)
npx expo prebuild --platform android --clean
cd android
.\gradlew.bat assembleRelease
```

## 📱 Mobile App Usage

1. **Launch the app** on your device
2. **Enter Server IP** - Your PC's local IP address (e.g., `192.168.1.100`)
3. **Enter ESP32 IP** - Your ESP32 camera's IP address (e.g., `192.168.1.50`)
4. **Tap Connect** - Establish connection with the server
5. **Control Features:**
   - **Flashlight**: Toggle between Auto/On/Off modes
   - **Motion**: Enable/disable motion detection
   - **Room Scan**: Start AI-powered object detection scan
   - **Result**: View scan results and use TTS to hear them

> **Note**: Ensure your mobile device and PC are on the same Wi-Fi network.

## 🔌 API Endpoints

### REST API

| Method | Endpoint | Description | Body |
|--------|----------|-------------|------|
| `POST` | `/connect` | Connect to ESP32 camera | `{ "ip": "192.168.1.50" }` |
| `POST` | `/flashlight` | Control flashlight | `{ "mode": "auto" \| "on" \| "off" }` |
| `POST` | `/motion` | Toggle motion detection | `{ "enabled": true \| false }` |
| `POST` | `/scan` | Start room scan | `{ "start": true, "duration": 10 }` |
| `GET` | `/status` | Get system status | - |

### WebSocket

- **Endpoint**: `ws://SERVER_IP:8000/ws`
- **Events**:
  - `scan_started` - Scan has begun
  - `scan_complete` - Scan finished with results
  - `motion` - Motion detected

## 🏗️ Project Structure

```
Aether-Eye-Controller/
├── hello.py                 # Python FastAPI server
├── requirements.txt         # Python dependencies
├── mobile/                  # React Native mobile app
│   ├── App.tsx             # Main app component
│   ├── app.json            # Expo configuration
│   ├── package.json        # Node dependencies
│   ├── eas.json            # EAS build configuration
│   ├── assets/             # App icons and images
│   └── scripts/             # Build scripts
└── README.md               # This file
```

## 🛠️ Technology Stack

### Server
- **FastAPI** - Modern Python web framework
- **YOLO (Ultralytics)** - Object detection model
- **OpenCV** - Computer vision processing
- **WebSocket** - Real-time communication
- **Faster Whisper** - Speech recognition (optional)

### Mobile App
- **React Native** - Cross-platform mobile framework
- **Expo** - Development platform
- **TypeScript** - Type-safe JavaScript
- **Expo Speech** - Text-to-speech functionality

## 📦 Building for Production

### Android APK

**Option 1: EAS Cloud Build** (Recommended)
```bash
cd mobile
npm run build:android
```

**Option 2: Local Build**
```bash
cd mobile
npx expo prebuild --platform android --clean
cd android
.\gradlew.bat assembleRelease
```

APK location: `android/app/build/outputs/apk/release/app-release.apk`

### iOS (macOS required)

```bash
cd mobile
npm run build:ios
```

## 🔧 Configuration

### Server Configuration (`hello.py`)

```python
CAM_SOURCE = "http://YOUR_ESP32_IP:81/stream"  # ESP32 camera URL
MIC_DEVICE_INDEX = 1                            # Microphone device
ASR_MODEL_SIZE = "small"                        # Whisper model size
YOLO_MODEL_PATH = "yolov8m.pt"                 # YOLO model path
```

### Mobile Configuration (`mobile/app.json`)

```json
{
  "expo": {
    "name": "Aether Mobile",
    "slug": "aether-mobile",
    "version": "1.0.0",
    "android": {
      "package": "com.aether.mobile",
      "permissions": ["INTERNET"]
    }
  }
}
```

## 🐛 Troubleshooting

### Server Issues
- **ESP32 not connecting**: Verify IP address and ensure ESP32 is on the same network
- **Camera stream not loading**: Check ESP32 camera stream URL format
- **YOLO model errors**: Ensure `yolov8m.pt` is in the project root

### Mobile App Issues
- **Connection failed**: Verify server IP and ensure both devices are on the same Wi-Fi
- **Build failures**: Check that all dependencies are installed (`npm install`)
- **WebSocket errors**: Ensure firewall allows connections on port 8000

## 📝 Development

### Running in Development Mode

**Server:**
```bash
python hello.py
```

**Mobile App:**
```bash
cd mobile
npm start
```

### Code Style

- Python: Follow PEP 8 guidelines
- TypeScript/React: Use ESLint and Prettier
- Commits: Follow conventional commit format

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**SypherKx**
- GitHub: [@SypherKx](https://github.com/SypherKx)
- Repository: [Aether-Eye-Controller](https://github.com/SypherKx/Aether-Eye-Controller)

## 🙏 Acknowledgments

- [YOLO](https://ultralytics.com/) for object detection
- [Expo](https://expo.dev/) for mobile app framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- ESP32 community for camera module support

## 📞 Support

For issues and questions:
- Open an [Issue](https://github.com/SypherKx/Aether-Eye-Controller/issues)
- Check existing [Discussions](https://github.com/SypherKx/Aether-Eye-Controller/discussions)

---

⭐ If you find this project useful, please give it a star!
