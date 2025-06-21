#!/usr/bin/env bash
set -euo pipefail

echo "Building test versions for update testing..."

# Create tar_files directory
mkdir -p tar_files

# Go to build directory
cd ../app

echo "=== Building v0.0.2 (clean build) ==="
bash build.sh dev ubuntu v0.0.2 --build-clean

echo "=== Copying v0.0.2 tar files ==="
cp ../output/*_v002.tar.gz ../tests/tar_files/

echo "=== Building v0.0.1 ==="
bash build.sh dev ubuntu v0.0.1

echo "=== Copying v0.0.1 tar files ==="
cp ../output/*_v001.tar.gz ../tests/tar_files/

echo "=== Build complete! ==="
echo "Files in tar_files:"
ls -la ../tests/tar_files/

echo ""
echo "Ready for testing with v0.0.1 dev environment installed."