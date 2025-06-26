#!/usr/bin/env bash

# Parse args
CLEAN=false
VERSION=""  # Add version parameter
ARGS=()
for arg in "$@"; do
    case $arg in
        --build-clean) CLEAN=true ;;
        --version=*) VERSION="${arg#*=}" ;;  # Extract version
        *) ARGS+=("$arg") ;;
    esac
done

TYPE=${ARGS[0]:-}
PLATFORM=${ARGS[1]:-ubuntu}

# Default version if not specified
if [[ -z "$VERSION" ]]; then
    VERSION="v0.0.1"
fi

# Special handling for 'run' command
if [[ "$TYPE" == "run" ]]; then
    EXTRA_ARGS=("${ARGS[@]:1}")
else
    EXTRA_ARGS=("${ARGS[@]:2}")
fi

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

##############################################################################
# Helper function to create info.json
##############################################################################
create_info_json() {
    local version=$1
    cat > info.json <<EOF
{"version": "$version"}
EOF
}

##############################################################################
# builders
##############################################################################
build_inference() {
    local PYI_ARGS="--onefile"
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

    PYI_ARGS="--onefile" 

    echo "Building 'inference' version $VERSION..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"

    # Create info.json for this build
    create_info_json "$VERSION" "inference"

    pyinstaller $PYI_ARGS \
        --hidden-import=urllib.request \
        --hidden-import=zipfile \
        --hidden-import=pyvips \
        --add-data "info.json:." \
        --name "$NAME" \
        --clean \
        --distpath "$DIST_DIR" \
        "$BOOTSTRAP"

    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$DIST_DIR"
    done
    
    # Create versioned tar file
    local VERSION_TAG="${VERSION//\./_}"  # v0.0.1 -> v0_0_1
    tar -czf "../output/inference_bootstrap_${VERSION_TAG}.tar.gz" -C "../output" "inference_bootstrap"
    
    # Also create unversioned for backward compatibility
    cp "../output/inference_bootstrap_${VERSION_TAG}.tar.gz" "../output/inference_bootstrap.tar.gz"
    
    # Clean up info.json
    rm -f info.json
    
    echo "✔ inference $VERSION → $DIST_DIR"
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

        PYI_ARGS="--onefile"
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

    echo "Building 'hypervisor' version $VERSION for $PLATFORM..."
    rm -rf "$DIST_DIR" "$SUP_DIR"; mkdir -p "$DIST_DIR" "$SUP_DIR"

    # Create info.json for this build
    create_info_json "$VERSION" "hypervisor"

    pyinstaller $PYI_ARGS \
        --hidden-import=urllib.request \
        --hidden-import=zipfile \
        --add-data "info.json:." \
        --name "$NAME" \
        --clean \
        --distpath "$DIST_DIR" \
        "$BOOTSTRAP"

    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$SUP_DIR/"
    done
    
    # Create versioned tar files
    local VERSION_TAG="${VERSION//\./_}"  # v0.0.1 -> v0_0_1
    tar -czf "../output/hypervisor_${VERSION_TAG}.tar.gz" -C "$SUP_DIR" .
    tar -czf "../output/moondream_station_ubuntu_${VERSION_TAG}.tar.gz" -C "$DIST_DIR" moondream_station
    
    # Also create unversioned for backward compatibility
    cp "../output/hypervisor_${VERSION_TAG}.tar.gz" "../output/hypervisor.tar.gz"
    cp "../output/moondream_station_ubuntu_${VERSION_TAG}.tar.gz" "../output/moondream_station_ubuntu.tar.gz"
    
    # Clean up info.json
    rm -f info.json
    
    echo "✔ hypervisor $VERSION → $DIST_DIR"
}

build_cli() {
    local NAME="moondream-cli"
    local DIST_DIR="../output/moondream-cli"
    local SRC_DIR="../app/moondream_cli"

    echo "Building 'cli' version $VERSION..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"
    
    # Create info.json for CLI (even though it's not compiled)
    create_info_json "$VERSION" "cli"
    
    cp -r "$SRC_DIR" "$DIST_DIR/"
    cp info.json "$DIST_DIR/moondream_cli/"
    
    # Create versioned tar file
    local VERSION_TAG="${VERSION//\./_}"  # v0.0.1 -> v0_0_1
    tar -czf "../output/moondream-cli_${VERSION_TAG}.tar.gz" -C "$DIST_DIR" moondream_cli
    
    # Also create unversioned for backward compatibility
    cp "../output/moondream-cli_${VERSION_TAG}.tar.gz" "../output/moondream-cli.tar.gz"
    
    # Clean up info.json
    rm -f info.json
    
    echo "✔ cli $VERSION → $DIST_DIR"
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

    echo "✔ dev sandbox ready → $DEV_DIR"
}

##############################################################################
# execution
##############################################################################
run_station() {
    cd ..
    ./output/moondream_station/moondream_station "$@"
}
##############################################################################
# dispatch
##############################################################################
case "$TYPE" in
    inference)   build_inference   ;;
    hypervisor)  build_hypervisor  ;;
    cli)         build_cli         ;;
    dev)         prepare_dev       ;;
    run)         run_station "${EXTRA_ARGS[@]}" ;;
    *)
        echo "Usage: $0 {inference|hypervisor|cli|dev} [platform] [--version=VERSION] | $0 run" >&2
        echo "Example: $0 hypervisor ubuntu --version=v0.0.2" >&2
        exit 1
        ;;
esac