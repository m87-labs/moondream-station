#!/usr/bin/env bash
set -euo pipefail

echo "Building test versions for update testing..."

mkdir -p tar_files

# Go to build directory
cd ../app

echo "=== Building v0.0.2 (clean build) with localhost manifest ==="
bash build.sh dev ubuntu v0.0.2 --build-clean --manifest-url "http://localhost:3020/manifest.json"

echo "=== Copying v0.0.2 tar files ==="
cp ../output/*_v002.tar.gz ../tests/tar_files/

echo "=== Building v0.0.1 with localhost manifest ==="
bash build.sh dev ubuntu v0.0.1 --manifest-url "http://localhost:3020/manifest.json"

echo "=== Copying v0.0.1 tar files ==="
cp ../output/*_v001.tar.gz ../tests/tar_files/

echo "=== Build complete! ==="
echo "Files in tar_files:"
ls -la ../tests/tar_files/

echo ""
echo "Ready for testing with v0.0.1 dev environment installed (pointing to localhost)."