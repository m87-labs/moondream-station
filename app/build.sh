#!/usr/bin/env bash

# Parse args
CLEAN=false
ARGS=()
for arg in "$@"; do
    case $arg in
        --build-clean) CLEAN=true ;;
        *) ARGS+=("$arg") ;;
    esac
done

TYPE=${ARGS[0]:-}
PLATFORM=${ARGS[1]:-ubuntu}

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

TYPE=${1:-}            # inference | hypervisor | cli | dev
PLATFORM=${2:-ubuntu}  # mac | ubuntu  (default ubuntu)

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

        # Build a single-file executable and drop libpython next to the bootstrap
        PYI_ARGS="--onefile" 
        #--add-binary ${LIBPYTHON}"

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
    tar -czf "../output/inference_bootstrap.tar.gz" -C "../output" "inference_bootstrap"
    echo "✔ inference → $DIST_DIR"
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
    tar -czf "../output/hypervisor.tar.gz" -C "$SUP_DIR" .
    tar -czf "../output/moondream_station_ubuntu.tar.gz" -C "$DIST_DIR" moondream_station
    echo "✔ hypervisor → $DIST_DIR"
}

build_cli() {
    local NAME="moondream-cli"
    local DIST_DIR="../output/moondream-cli"
    local SRC_DIR="../app/moondream_cli"

    echo "Building 'cli'..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"
    cp -r "$SRC_DIR" "$DIST_DIR/"
    tar -czf "../output/moondream-cli.tar.gz" -C "$DIST_DIR" moondream_cli
    echo "✔ cli → $DIST_DIR"
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
    run)         run_station "$@"  ;;
    *)
        echo "Usage: $0 {inference|hypervisor|cli|dev} [platform] | $0 run [options]" >&2
        echo "Options:" >&2
        echo "  --build-clean           Clean output and dev directories before building" >&2
        echo "  --manifest-url URL      Set custom manifest URL (overrides default)" >&2
        exit 1
        ;;
esac