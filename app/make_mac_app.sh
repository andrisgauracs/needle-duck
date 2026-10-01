#!/usr/bin/env bash
# Builds "Needle Duck Studio.app" in the repo root: a double-clickable launcher for app/main.py.
# Run ./setup.sh first (the launcher uses .venv). Rerun this if you move the repo.
set -e
cd "$(dirname "$0")/.."
A="Needle Duck Studio.app"
rm -rf "$A"
mkdir -p "$A/Contents/MacOS" "$A/Contents/Resources"
cp app/assets/icon.icns "$A/Contents/Resources/icon.icns"
cat > "$A/Contents/MacOS/NeedleDuckStudio" <<'SH'
#!/bin/bash
# Launches the studio from the repo folder this bundle sits in.
DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$DIR"
exec "$DIR/.venv/bin/python" "$DIR/app/main.py" >> "$DIR/app/studio.log" 2>&1
SH
chmod +x "$A/Contents/MacOS/NeedleDuckStudio"
cat > "$A/Contents/Info.plist" <<'PL'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Needle Duck Studio</string>
  <key>CFBundleDisplayName</key><string>Needle Duck Studio</string>
  <key>CFBundleIdentifier</key><string>com.betterstack.needleduckstudio</string>
  <key>CFBundleExecutable</key><string>NeedleDuckStudio</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSLocalNetworkUsageDescription</key><string>Needle Duck Studio sends training data to your GPU fine-tuning server on the local network.</string>
</dict></plist>
PL
# an ad-hoc signature lets macOS show the Local Network permission prompt for this app
codesign --force --deep -s - "$A" >/dev/null
echo "built $A"
