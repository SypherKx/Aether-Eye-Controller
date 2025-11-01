/*
 Generates placeholder PNG icons so Expo prebuild won’t fail.
 Creates ./assets/icon.png and ./assets/adaptive-icon.png (1024x1024).
*/

const { PNG } = require('pngjs');
const fs = require('fs');
const path = require('path');

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function createSolidPng(width, height, rgba) {
  const png = new PNG({ width, height });
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (width * y + x) << 2;
      png.data[idx] = rgba[0];
      png.data[idx + 1] = rgba[1];
      png.data[idx + 2] = rgba[2];
      png.data[idx + 3] = rgba[3];
    }
  }
  return png;
}

function writePng(png, outPath) {
  return new Promise((resolve, reject) => {
    const stream = fs.createWriteStream(outPath);
    png.pack().pipe(stream);
    stream.on('finish', resolve);
    stream.on('error', reject);
  });
}

async function main() {
  const assetsDir = path.join(process.cwd(), 'assets');
  ensureDir(assetsDir);

  const iconPath = path.join(assetsDir, 'icon.png');
  const adaptivePath = path.join(assetsDir, 'adaptive-icon.png');

  const icon = createSolidPng(1024, 1024, [28, 28, 28, 255]); // dark gray
  const adaptive = createSolidPng(1024, 1024, [255, 255, 255, 255]); // white

  await writePng(icon, iconPath);
  await writePng(adaptive, adaptivePath);

  console.log('Generated:', iconPath);
  console.log('Generated:', adaptivePath);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});




