#!/bin/bash
# Build "YuE Studio.app" and a .dmg. Run from anywhere: bash app/package.sh
# Bundles: the Swift app, the uv installer binary, and the patched YuE source with the prebuilt
# Neural Engine library. Everything else (Python, packages, model weights) is installed on first launch.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APPDIR="$ROOT/app/YuEStudio"
DIST="$ROOT/dist"
APP="$DIST/YuE Studio.app"
APPICON="$ROOT/app/YuEStudio/Sources/YuEStudio/Resources/AppIcon.icon"
DMG="$DIST/YuE-Studio.dmg"
UV="${UV:-$(command -v uv)}"
VERSION="$(date +%Y%m%d)-$(cat "$ROOT"/src/yue2/*.py "$ROOT"/src/yue2/ane/*.py "$ROOT"/tools/yue2_worker.py "$ROOT"/tools/transcribe_sheetsage.py "$ROOT"/tools/download_models.py "$ROOT"/pyproject.toml | shasum | cut -c1-8)"

if [ -e "$APP" ]; then
  echo "== removing previously built $APP"
  rm -rf "$APP"
fi

if [ -e "$DMG" ]; then
  echo "== removing previously built $DMG"
  rm -rf "$DMG"
fi

echo "== building Neural Engine library"
(cd "$ROOT/src/yue2/ane" && clang -O2 -fobjc-arc -dynamiclib libyue2ane.m -o libyue2ane.dylib -framework Foundation -framework IOSurface)
echo "== building app ($VERSION)"
(cd "$APPDIR" && swift build -c release 2>&1 | grep -E "error|Build complete")

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/payload/yue2-src"
cp "$APPDIR/.build/release/YuEStudio" "$APP/Contents/MacOS/YuE Studio"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>YuE Studio</string>
  <key>CFBundleDisplayName</key>
  <string>YuE Studio</string>
  <key>CFBundleIdentifier</key>
  <string>com.tonyweston.yuestudio</string>
  <key>CFBundleExecutable</key>
  <string>YuE Studio</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleVersion</key>
  <string>$VERSION</string>
  <key>CFBundleShortVersionString</key>
  <string>0.5.0</string>
  <key>CFBundleIconFile</key>
	<string>AppIcon</string>
	<key>CFBundleIconName</key>
	<string>AppIcon</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>LSArchitecturePriority</key>
  <array>
    <string>arm64</string>
  </array>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSBonjourServices</key><array>
  <string>_yuestudio._tcp</string></array>
  <key>NSMicrophoneUsageDescription</key>
  <string>YuE Studio records a hummed melody to turn it into a song.</string>
  <key>NSLocalNetworkUsageDescription</key>
  <string>YuE Studio looks for an iPhone running YuE Remote to use its Neural Engine for synthesis.</string>
  <key>NSHumanReadableCopyright</key>
  <string>YuE2 model: m-a-p (Apache-2.0 code, model licence applies). SheetSage2 weights: CC BY-NC 4.0.</string>
</dict>
</plist>
PLIST

echo "== payload"
cp "$UV" "$APP/Contents/Resources/payload/uv"
echo "$VERSION" > "$APP/Contents/Resources/payload/version.txt"
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/pyproject.toml" "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/MODEL_LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$ROOT/MANIFEST.in" "$ROOT/licenses" \
  "$ROOT/src" "$ROOT/examples" "$APP/Contents/Resources/payload/yue2-src/"
mkdir -p "$APP/Contents/Resources/payload/yue2-src/tools"
cp "$ROOT/tools/yue2_worker.py" "$ROOT/tools/download_models.py" "$ROOT/tools/transcribe_sheetsage.py" "$APP/Contents/Resources/payload/yue2-src/tools/"

echo "== app icon"

xcrun actool "${APPICON}" \
    --compile "${APP}/Contents/Resources" \
    --app-icon AppIcon \
    --platform macosx \
    --target-device mac \
    --minimum-deployment-target 15.0 \
    --output-partial-info-plist "/tmp/iconassetcatalog_generated_info.plist" \
    2> /dev/null > /dev/null

# Signing. Ad hoc by default; SIGN_IDENTITY='Developer ID Application: Name (TEAM)' signs for distribution
# with the hardened runtime and secure timestamps, nested binaries first (notarization requires all of them).
SIGN="${SIGN_IDENTITY:--}"
if [ "$SIGN" = "-" ]; then
  echo "== signing (ad hoc)"; codesign --force --deep --sign - "$APP"
else
  echo "== signing with $SIGN"
  for bin in "$APP/Contents/Resources/payload/uv" "$APP/Contents/Resources/payload/yue2-src/src/yue2/ane/libyue2ane.dylib"; do
    codesign --force --options runtime --timestamp --sign "$SIGN" "$bin"
  done
  codesign --force --options runtime --timestamp --entitlements "$APPDIR/YuEStudio.entitlements" --sign "$SIGN" "$APP"
  codesign --verify --deep --strict "$APP" && echo "signature verified"
fi

echo "== disk image"
hdiutil create "$DMG" -volname "YuE Studio" -srcfolder "$APP" -ov -format UDZO -quiet
[ "$SIGN" != "-" ] && codesign --force --timestamp --sign "$SIGN" "$DMG"

# Notarization: NOTARY_PROFILE=AC_PROFILE (stored with `xcrun notarytool store-credentials`).
# The app is notarized first (as a zip) and stapled, then the final image is built, notarized and stapled,
# so both carry tickets and open on other Macs with no warning even offline.
if [ -n "${NOTARY_PROFILE:-}" ]; then
  echo "== notarizing the app (a few minutes)"
  ditto -c -k --keepParent "$APP" "$DIST/YuE-Studio.zip"
  xcrun notarytool submit "$DIST/YuE-Studio.zip" --keychain-profile "$NOTARY_PROFILE" --wait
  rm -f "$DIST/YuE-Studio.zip"
  xcrun stapler staple "$APP"
  rm -f "$DMG"
  hdiutil create "$DMG" -volname "YuE Studio" -srcfolder "$APP" -ov -format UDZO -quiet
  codesign --force --timestamp --sign "$SIGN" "$DMG"
  echo "== notarizing the disk image"
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
  spctl --assess --type open --context context:primary-signature "$DMG" && echo "Gatekeeper accepts the image"
  spctl --assess --type execute "$APP" && echo "Gatekeeper accepts the app"
fi
du -sh "$APP" "$DMG"
echo "done: $APP"
