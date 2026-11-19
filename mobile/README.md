# 📱 CrowdSense Mobile — Cross-Platform AI Crowd Counter

A cross-platform **Flutter & Dart** mobile application for real-time crowd density sensing, headcount estimation, and spatial congestion analysis powered by the CrowdSense neural backend.

---

## 🌟 Key Features

1. **Instant Camera & Gallery Crowd Scanning:**
   * Take photos directly using your phone camera or select from your gallery.
   * Transmits to the CrowdSense inference engine for sub-second deep density estimation.
2. **Interactive Heatmap Overlay:**
   * View the original crowd scene with a smoothly blended glowing density heatmap overlay.
   * Real-time opacity slider ($0.0 \rightarrow 1.0$) to inspect individual human head locations.
3. **Interactive ROI Zone Counting:**
   * Tap and drag a bounding box on the image directly with your finger to calculate the headcount within that specific zone.
4. **Surveillance Video Analytics:**
   * Upload CCTV or security clips (MP4/AVI) to generate a temporal headcount timeline chart with peak crowd count and average density statistics.
5. **Real-time Diagnostics & Uncertainty:**
   * Displays 95% Confidence Interval ($\bar{C} \pm \Delta$), crowd risk category (Low, Moderate, High, Critical), latency, and active architecture (MCNN / CSRNet).
6. **Configurable Cloud & Local Backend:**
   * Easily configure the backend URL in Settings (e.g., your Render deployment `https://crowdsense.onrender.com` or local `http://10.0.2.2:5000`).

---

## 🎨 Branding & Logo
* **Master High-Res Logo:** Located at `assets/images/logo.png` (1024x1024).
* **Launcher Icons:** Multi-density Android mipmaps generated across `mdpi`, `hdpi`, `xhdpi`, `xxhdpi`, and `xxxhdpi` in `android/app/src/main/res/`.

---

## 🚀 How to Run the App

### Prerequisites
* Flutter SDK (3.47+ recommended)
* Android Studio / Xcode or a connected mobile device / emulator

### 1. Run in Development Mode
Connect your Android phone via USB (with USB Debugging enabled) or start an emulator:
```bash
cd mobile
flutter run
```

### 2. Build Android APK
To generate a standalone APK that can be installed on any Android phone:
```bash
cd mobile
flutter build apk --debug
# Output: mobile/build/app/outputs/flutter-apk/app-debug.apk
```

For production release APK:
```bash
flutter build apk --release
# Output: mobile/build/app/outputs/flutter-apk/app-release.apk
```
