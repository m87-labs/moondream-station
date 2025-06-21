#!/usr/bin/env bash
# Usage: ./build.sh <type> <platform> <version> [--manifest-url URL]

# Parse args more carefully
CLEAN=false
MANIFEST_URL=""
ARGS=()

# First pass: extract flags and build clean args array
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-clean)
            CLEAN=true
            shift
            ;;
        --manifest-url=*)
            MANIFEST_URL="${1#*=}"
            shift
            ;;
        --manifest-url)
            shift
            MANIFEST_URL="$1"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

TYPE=${ARGS[0]:-}
PLATFORM=${ARGS[1]:-ubuntu}
VERSION=${ARGS[2]:-v0.0.1}

echo "Building with version: $VERSION"
if [ -n "$MANIFEST_URL" ]; then
    echo "Using custom manifest URL: $MANIFEST_URL"
fi

update_version_strings() {
    local version=$1
    local manifest_url=$2
    echo "Updating version strings to $version..."
    
    # Update bootstrap version (this is the main executable)
    sed -i.bak "s/BOOTSTRAP_VERSION = \".*\"/BOOTSTRAP_VERSION = \"$version\"/" ../app/hypervisor/bootstrap.py
    
    # Update hypervisor version (if it exists)
    if grep -q "HYPERVISOR_VERSION" ../app/hypervisor/hypervisor.py; then
        sed -i.bak "s/HYPERVISOR_VERSION = \".*\"/HYPERVISOR_VERSION = \"$version\"/" ../app/hypervisor/hypervisor.py
    fi
    
    # Update inference version  
    sed -i.bak "s/VERSION = \".*\"/VERSION = \"$version\"/" ../app/inference_client/main.py
    
    # Update CLI version (check if __init__.py exists, if not try other locations)
    if [ -f "../app/moondream_cli/__init__.py" ]; then
        sed -i.bak "s/VERSION = \".*\"/VERSION = \"$version\"/" ../app/moondream_cli/__init__.py
    elif [ -f "../app/moondream_cli/cli.py" ]; then
        sed -i.bak "s/VERSION = \".*\"/VERSION = \"$version\"/" ../app/moondream_cli/cli.py
    else
        echo "Warning: Could not find CLI version file"
    fi
    
    # Update manifest URL if provided
    if [ -n "$manifest_url" ]; then
        echo "Updating manifest URL to: $manifest_url"
        echo "Before update:"
        grep "MANIFEST_URL" ../app/hypervisor/manifest.py
        sed -i.bak "s|MANIFEST_URL = \".*\"|MANIFEST_URL = \"$manifest_url\"|" ../app/hypervisor/manifest.py
        echo "After update:"
        grep "MANIFEST_URL" ../app/hypervisor/manifest.py
    fi
}

restore_version_strings() {
    echo "Restoring original version strings..."
    # Restore from .bak files
    find ../app -name "*.bak" -exec sh -c 'mv "$1" "${1%.bak}"' _ {} \;
}

# Set up cleanup trap
trap restore_version_strings EXIT

if $CLEAN; then
    echo "Cleaning output and dev directories..."
    rm -rf ../output
    if [[ "$PLATFORM" = "mac" ]]; then
        rm -rf "$HOME/Library/MoondreamStation"
    elif [[ "$PLATFORM" = "ubuntu" ]]; then
        rm -rf "$HOME/.local/share/MoondreamStation"
    fi
fi

set -euo pipefail

# Update version strings before building
update_version_strings "$VERSION" "$MANIFEST_URL"

##############################################################################
# builders
##############################################################################
build_inference() {
    local NAME="inference_bootstrap"
    local DIST_DIR="../output/inference_bootstrap"
    local BOOTSTRAP="../app/inference_client/bootstrap.py"
    local SRC_DIR="../app/inference_client"
    local FILES=(main.py model_service.py requirements.txt)
    local LIBPYTHON
        LIBPYTHON=$(
python - <<'PY'
import os, sysconfig, pathlib, sys
libdir = pathlib.Path(sysconfig.get_config_var("LIBDIR") or "")
ver     = sysconfig.get_config_var("LDVERSION")          # e.g. 3.10
wanted  = libdir / f"libpython{ver}.so.1.0"              # soname
# fall back to the un-versioned file only if the soname is missing
for p in (wanted, libdir / f"libpython{ver}.so"):
    if p.exists():
        # always *install* it under the soname so the boot-loader sees it
        print(f"{p}{os.pathsep}libpython{ver}.so.1.0")
        break
else:
    sys.stderr.write("ERROR: libpython with --enable-shared not found\n")
    sys.exit(1)
PY
) || exit 1
        
        # Build with libpython bundled
        local PYI_ARGS="--onefile --add-binary ${LIBPYTHON}"
        
    echo "Building 'inference'..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"
    pyinstaller $PYI_ARGS \
        --hidden-import=urllib.request \
        --hidden-import=zipfile \
        --hidden-import=pyvips \
        --name "$NAME" \
        --clean \
        --distpath "$DIST_DIR" \
        "$BOOTSTRAP"
    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$DIST_DIR"
    done
    echo "!!!!!! my current working dir is $PWD"
    
    # Create versioned tarball
    local VERSION_SUFFIX="_${VERSION//./}"  # v0.0.1 -> _v001
    tar -czf "../output/inference_bootstrap${VERSION_SUFFIX}.tar.gz" -C "../output" "inference_bootstrap"
    
    echo "✔ inference → $DIST_DIR (${VERSION})"
}

build_hypervisor() {
    local PYI_ARGS
    if [[ "$PLATFORM" = "mac" ]]; then
        # macOS always embeds Python.framework for us
        PYI_ARGS="--windowed"
    elif [[ "$PLATFORM" = "ubuntu" ]]; then
        # Issue with ubuntu and the py .so, bundling it. Should also let us run on <Ubuntu 22
        local LIBPYTHON
        LIBPYTHON=$(
python - <<'PY'
import os, sysconfig, pathlib, sys
libdir = pathlib.Path(sysconfig.get_config_var("LIBDIR") or "")
ver     = sysconfig.get_config_var("LDVERSION")          # e.g. 3.10
wanted  = libdir / f"libpython{ver}.so.1.0"              # soname
# fall back to the un-versioned file only if the soname is missing
for p in (wanted, libdir / f"libpython{ver}.so"):
    if p.exists():
        # always *install* it under the soname so the boot-loader sees it
        print(f"{p}{os.pathsep}libpython{ver}.so.1.0")
        break
else:
    sys.stderr.write("ERROR: libpython with --enable-shared not found\n")
    sys.exit(1)
PY
) || exit 1
        # Build a single-file executable and drop libpython next to the bootstrap
        PYI_ARGS="--onefile"
        # --add-binary ${LIBPYTHON}"
    else
        echo "Unknown platform '$PLATFORM' (mac|ubuntu)" >&2
        exit 1
    fi
    local NAME="moondream_station"
    local DIST_DIR="../output/moondream_station"
    local SUP_DIR="../output/moondream-station-files"
    local BOOTSTRAP="../app/hypervisor/bootstrap.py"
    local SRC_DIR="../app/hypervisor"
    local FILES=(
        hypervisor_server.py hypervisor.py inferencevisor.py requirements.txt
        manifest.py config.py misc.py update_bootstrap.sh clivisor.py
        display_utils.py
    )
    echo "Building 'hypervisor' for $PLATFORM..."
    rm -rf "$DIST_DIR" "$SUP_DIR"; mkdir -p "$DIST_DIR" "$SUP_DIR"
    pyinstaller $PYI_ARGS \
        --hidden-import=urllib.request \
        --hidden-import=zipfile \
        --name "$NAME" \
        --clean \
        --distpath "$DIST_DIR" \
        "$BOOTSTRAP"
    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$SUP_DIR/"
    done
    
    # Create versioned tarballs
    local VERSION_SUFFIX="_${VERSION//./}"  # v0.0.1 -> _v001
    tar -czf "../output/hypervisor${VERSION_SUFFIX}.tar.gz" -C "$SUP_DIR" .
    tar -czf "../output/moondream_station_ubuntu${VERSION_SUFFIX}.tar.gz" -C "$DIST_DIR" moondream_station
    
    echo "✔ hypervisor → $DIST_DIR (${VERSION})"
}

build_cli() {
    local NAME="moondream-cli"
    local DIST_DIR="../output/moondream-cli"
    local SRC_DIR="../app/moondream_cli"
    echo "Building 'cli'..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"
    cp -r "$SRC_DIR" "$DIST_DIR/"
    
    # Create versioned tarball
    local VERSION_SUFFIX="_${VERSION//./}"  # v0.0.1 -> _v001
    tar -czf "../output/moondream-cli${VERSION_SUFFIX}.tar.gz" -C "$DIST_DIR" moondream_cli
    
    echo "✔ cli → $DIST_DIR (${VERSION})"
}

##############################################################################
# dev sandbox
##############################################################################
prepare_dev() {
    build_cli
    build_inference
    build_hypervisor
    local DEV_DIR
    if [[ "$PLATFORM" = "mac" ]]; then
        DEV_DIR="$HOME/Library/MoondreamStation"
    elif [[ "$PLATFORM" = "ubuntu" ]]; then
        DEV_DIR="$HOME/.local/share/MoondreamStation"
    else
        echo "Unknown platform '$PLATFORM' (mac|ubuntu)" >&2; exit 1
    fi
    mkdir -p "$DEV_DIR/inference/v0.0.1"
    # copy hypervisor supplements
    local HYP_SRC="../output/moondream-station-files"
    local HYP_FILES=(
        hypervisor_server.py hypervisor.py inferencevisor.py requirements.txt
        manifest.py config.py misc.py update_bootstrap.sh clivisor.py
        display_utils.py
    )
    for f in "${HYP_FILES[@]}"; do
        cp "$HYP_SRC/$f" "$DEV_DIR/"
    done
    # copy CLI dir
    cp -r "../output/moondream-cli/moondream_cli" "$DEV_DIR/"
    # copy inference build
    cp -r "../output/inference_bootstrap" "$DEV_DIR/inference/v0.0.1/"
    echo "✔ dev sandbox ready → $DEV_DIR (${VERSION})"
}

##############################################################################
# execution
##############################################################################
run_station() {
    cd ..
    ./output/moondream_station/moondream_station
}

##############################################################################
# dispatch
##############################################################################
case "$TYPE" in
    inference)   build_inference   ;;
    hypervisor)  build_hypervisor  ;;
    cli)         build_cli         ;;
    dev)         prepare_dev       ;;
    run)         run_station       ;;
    *)
        echo "Usage: $0 {inference|hypervisor|cli|dev} [platform] [version] [--manifest-url URL] | $0 run" >&2
        echo "Options:" >&2
        echo "  --build-clean           Clean output and dev directories before building" >&2
        echo "  --manifest-url URL      Set custom manifest URL (overrides default)" >&2
        exit 1
        ;;
esac