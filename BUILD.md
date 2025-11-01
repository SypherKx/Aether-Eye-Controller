# Building Android APK

## Fixed Issues
✅ Created missing `index.js` entry point
✅ Added INTERNET permission for Android
✅ Fixed WebSocket URL construction
✅ Improved error handling in App.tsx
✅ Added proper cleanup for WebSocket connections
✅ Updated EAS configuration

## Build Methods

### Method 1: EAS Cloud Build (Recommended)
```bash
cd mobile
npm run build:android
```

This will build an APK in the cloud. Once complete, download it from:
https://expo.dev/accounts/sypherkx/projects/aether-mobile/builds

### Method 2: Local Build (Requires Android Studio)
If you have Android Studio and Java installed:

1. Generate native Android project:
```bash
cd mobile
npx expo prebuild --platform android --clean
```

2. Build APK:
```bash
cd android
.\gradlew.bat assembleRelease
```

The APK will be in: `android/app/build/outputs/apk/release/app-release.apk`

### Method 3: Using Android Studio
1. Run `npx expo prebuild --platform android --clean`
2. Open `android` folder in Android Studio
3. Build > Build Bundle(s) / APK(s) > Build APK(s)

## Requirements
- Node.js and npm installed
- For local builds: Java JDK and Android Studio
- For cloud builds: EAS account (already configured)

## Troubleshooting
If EAS build fails, check the build logs at the provided URL for specific errors.
Most common issues:
- Missing dependencies (run `npm install`)
- Invalid icon files (check `assets/icon.png` and `assets/adaptive-icon.png`)
- Configuration errors in `app.json`

