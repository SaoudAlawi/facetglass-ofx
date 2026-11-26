#!/bin/sh
# Installs the built FacetGlass.ofx.bundle into the system OFX plugin folder.
# Needs admin rights (asks for your password via sudo).
#   sh tools/install_macos.sh [build]
set -e
BUILD="${1:-build}"
SRC="$BUILD/FacetGlass.ofx.bundle"
DST_ROOT="/Library/OFX/Plugins"
DST="$DST_ROOT/FacetGlass.ofx.bundle"

if [ ! -f "$SRC/Contents/MacOS/FacetGlass.ofx" ]; then
    echo "No built plugin at $SRC/Contents/MacOS/FacetGlass.ofx - build first." >&2
    exit 1
fi
if pgrep -x Resolve >/dev/null 2>&1; then
    echo "Warning: DaVinci Resolve is running and keeps the old plugin loaded. Quit and restart it after installing."
fi

# Stage outside protected folders (Desktop/Documents): macOS privacy can stop
# root from reading them, which silently leaves an empty bundle.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$SRC" "$STAGE/"
chmod -R a+rX "$STAGE"

# Replace only our own bundle folder.
sudo mkdir -p "$DST_ROOT"
sudo rm -rf "$DST"
sudo cp -R "$STAGE/FacetGlass.ofx.bundle" "$DST"
sudo xattr -dr com.apple.quarantine "$DST" 2>/dev/null || true
codesign --verify "$DST" && echo "Installed $DST ($(stat -f '%z bytes, %Sm' "$DST/Contents/MacOS/FacetGlass.ofx"))"
