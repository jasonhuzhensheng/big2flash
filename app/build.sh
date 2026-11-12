#!/bin/sh
# Builds big2flash.app (menu-bar app) into app/build/, bundling the engine scripts from ../engine.
set -e
cd "$(dirname "$0")"
APP=build/big2flash.app
rm -rf "$APP" build/icon.iconset
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/engine" build/icon.iconset

swiftc -O -parse-as-library -swift-version 5 -target arm64-apple-macos14 big2flash.swift -o "$APP/Contents/MacOS/big2flash"
cp Info.plist "$APP/Contents/"
cp ../engine/server.py ../engine/mtp.py ../engine/flash_next.py ../engine/moe_kernel.py ../engine/dense_kernel.py \
   ../engine/claude-local.sh "$APP/Contents/Resources/engine/"
cp requirements.txt "$APP/Contents/Resources/"

swiftc -O make_icon.swift -o build/make_icon
build/make_icon build/icon-1024.png
for s in 16 32 128 256 512; do
  sips -z $s $s build/icon-1024.png --out build/icon.iconset/icon_${s}x${s}.png >/dev/null
  sips -z $((s * 2)) $((s * 2)) build/icon-1024.png --out build/icon.iconset/icon_${s}x${s}@2x.png >/dev/null
done
iconutil -c icns build/icon.iconset -o "$APP/Contents/Resources/AppIcon.icns"

codesign --force --deep -s - "$APP"
echo "built $APP"
