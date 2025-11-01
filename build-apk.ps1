# Quick Android APK Build Script
Write-Host "Building Android APK..." -ForegroundColor Green

# Check if Java is available
$javaCheck = Get-Command java -ErrorAction SilentlyContinue
if (-not $javaCheck) {
    Write-Host "Java not found. Please install JDK 17 or later." -ForegroundColor Yellow
    Write-Host "Alternatively, use EAS cloud build: npm run build:android" -ForegroundColor Yellow
    exit 1
}

# Build APK
cd android
.\gradlew.bat assembleRelease

if ($LASTEXITCODE -eq 0) {
    $apkPath = "app\build\outputs\apk\release\app-release.apk"
    if (Test-Path $apkPath) {
        Write-Host "`n✅ APK built successfully!" -ForegroundColor Green
        Write-Host "Location: $PWD\$apkPath" -ForegroundColor Cyan
        Write-Host "`nYou can install this APK on your Android device." -ForegroundColor Yellow
    }
} else {
    Write-Host "`n❌ Build failed. Check errors above." -ForegroundColor Red
}

