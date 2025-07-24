#!/usr/bin/env bash

# Parse args
CLEAN=false
HYPERVISOR_VERSION=""
CLI_VERSION=""
INFERENCE_VERSION=""
BOOTSTRAP_VERSION=""
ARGS=()
for arg in "$@"; do
    case $arg in
        --build-clean) CLEAN=true ;;
        --hypervisor-version=*) HYPERVISOR_VERSION="${arg#*=}" ;;
        --cli-version=*) CLI_VERSION="${arg#*=}" ;;
        --inference-version=*) INFERENCE_VERSION="${arg#*=}" ;;
        --bootstrap-version=*) BOOTSTRAP_VERSION="${arg#*=}" ;;
        --version=*) # Legacy: sets all versions
            VERSION="${arg#*=}"
            HYPERVISOR_VERSION="$VERSION"
            CLI_VERSION="$VERSION"
            INFERENCE_VERSION="$VERSION"
            BOOTSTRAP_VERSION="$VERSION" ;;
        *) ARGS+=("$arg") ;;
    esac
done

TYPE=${ARGS[0]:-}
PLATFORM=${ARGS[1]:-ubuntu}

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
    local component=$2
    local build_date=$(date +%Y-%m-%d)
    
    cat > info.json <<EOF
{
    "version": "$version",
    "component": "$component",
    "build_date": "$build_date",
    "platform": "$PLATFORM"
}
EOF
}

##############################################################################
# builders
##############################################################################
build_inference() {
    local VERSION="$INFERENCE_VERSION"
    local PYI_ARGS="--onefile"
    local NAME="inference_bootstrap"
    local DIST_DIR="../output/inference_bootstrap"
    local BOOTSTRAP="../app/inference_client/bootstrap.py"
    local SRC_DIR="../app/inference_client"
    local FILES=(main.py model_service.py requirements.txt)

#     local LIBPYTHON
#     LIBPYTHON=$(
# python - <<'PY'
# import os, sysconfig, pathlib, sys
# libdir = pathlib.Path(sysconfig.get_config_var("LIBDIR") or "")
# ver     = sysconfig.get_config_var("LDVERSION")          # e.g. 3.10
# wanted  = libdir / f"libpython{ver}.so.1.0"              # soname
# # fall back to the un-versioned file only if the soname is missing
# for p in (wanted, libdir / f"libpython{ver}.so"):
#     if p.exists():
#         # always *install* it under the soname so the boot-loader sees it
#         print(f"{p}{os.pathsep}libpython{ver}.so.1.0")
#         break
# else:
#     sys.stderr.write("ERROR: libpython with --enable-shared not found\n")
#     sys.exit(1)
# PY
# ) || exit 1

    PYI_ARGS="--onefile" 

    echo "Building 'inference'..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"

    # Only create info.json and add to build if version was specified
    local ADD_DATA_ARG=""
    if [[ -n "$VERSION" ]]; then
        create_info_json "$VERSION" "inference"
        ADD_DATA_ARG="--add-data info.json:."
    fi

    pyinstaller $PYI_ARGS \
        --hidden-import=urllib.request \
        --hidden-import=zipfile \
        --hidden-import=pyvips \
        $ADD_DATA_ARG \
        --name "$NAME" \
        --clean \
        --distpath "$DIST_DIR" \
        "$BOOTSTRAP"

    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$DIST_DIR"
    done
    
    # Create tar file
    tar -czf "../output/inference_bootstrap.tar.gz" -C "../output" "inference_bootstrap"
    
    # Clean up info.json if it exists
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

    echo "Building 'hypervisor' for $PLATFORM..."
    rm -rf "$DIST_DIR" "$SUP_DIR"; mkdir -p "$DIST_DIR" "$SUP_DIR"
    
    # Bootstrap version for the moondream_station executable
    local ADD_DATA_ARG=""
    if [[ -n "$BOOTSTRAP_VERSION" ]]; then
        echo "  bootstrap version $BOOTSTRAP_VERSION"
        create_info_json "$BOOTSTRAP_VERSION" "bootstrap"
        ADD_DATA_ARG="--add-data info.json:."
    fi
    
    pyinstaller $PYI_ARGS \
        --hidden-import=urllib.request \
        --hidden-import=zipfile \
        $ADD_DATA_ARG \
        --name "$NAME" \
        --clean \
        --distpath "$DIST_DIR" \
        "$BOOTSTRAP"
    
    # Clean up bootstrap info.json
    rm -f info.json
    
    for f in "${FILES[@]}"; do
        cp "$SRC_DIR/$f" "$SUP_DIR/"
    done
    
    # Hypervisor version for the component files
    if [[ -n "$HYPERVISOR_VERSION" ]]; then
        echo "  hypervisor component version $HYPERVISOR_VERSION"
        create_info_json "$HYPERVISOR_VERSION" "hypervisor"
        cp info.json "$SUP_DIR/"
        rm -f info.json
    fi
    
    # Create versioned tar files
    tar -czf "../output/hypervisor.tar.gz" -C "$SUP_DIR" .
    tar -czf "../output/moondream_station_ubuntu.tar.gz" -C "$DIST_DIR" moondream_station
    
    echo "✔ hypervisor (bootstrap: $BOOTSTRAP_VERSION, components: $HYPERVISOR_VERSION) → $DIST_DIR"
}

build_cli() {
    local VERSION="$CLI_VERSION"
    local NAME="moondream-cli"
    local DIST_DIR="../output/moondream-cli"
    local SRC_DIR="../app/moondream_cli"

    echo "Building 'cli'..."
    rm -rf "$DIST_DIR"; mkdir -p "$DIST_DIR"
    
    cp -r "$SRC_DIR" "$DIST_DIR/"
    
    # Only create and copy info.json if version was specified
    if [[ -n "$VERSION" ]]; then
        create_info_json "$VERSION" "cli"
        cp info.json "$DIST_DIR/moondream_cli/"
        rm -f info.json
    fi
    
    # Create unversioned tar file
    tar -czf "../output/moondream-cli.tar.gz" -C "$DIST_DIR" moondream_cli
    
    echo "✔ cli $VERSION → $DIST_DIR"
}

##############################################################################
# dev sandbox
##############################################################################
prepare_dev() {
    # Build all components
    echo "Building dev environment..."
    
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
    # For inference directory, use specified version or default
    local INFERENCE_VERSION="${INFERENCE_VERSION:-v0.0.1}"
    mkdir -p "$DEV_DIR/inference/$INFERENCE_VERSION"

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
    
    # Only copy info.json if it exists
    if [[ -f "$HYP_SRC/info.json" ]]; then
        cp "$HYP_SRC/info.json" "$DEV_DIR/"
    fi

    # copy CLI dir
    cp -r "../output/moondream-cli/moondream_cli" "$DEV_DIR/"

    # copy inference build to versioned directory
    cp -r "../output/inference_bootstrap" "$DEV_DIR/inference/$INFERENCE_VERSION/"

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
        echo "Usage: $0 {inference|hypervisor|cli|dev} [platform] [options] | $0 run" >&2
        echo "Options:" >&2
        echo "  --hypervisor-version=VERSION  Set hypervisor component version" >&2
        echo "  --cli-version=VERSION        Set CLI version" >&2
        echo "  --inference-version=VERSION  Set inference version" >&2
        echo "  --bootstrap-version=VERSION  Set bootstrap (moondream_station executable) version" >&2
        echo "  --version=VERSION           Set all component versions" >&2
        echo "  --build-clean               Clean before building" >&2
        echo "" >&2
        echo "Examples:" >&2
        echo "  $0 hypervisor ubuntu --hypervisor-version=v0.0.2" >&2
        echo "  $0 hypervisor ubuntu --bootstrap-version=v0.0.2 --hypervisor-version=v0.0.1" >&2
        echo "  $0 dev ubuntu --hypervisor-version=v0.0.1 --cli-version=v0.0.2 --inference-version=v0.0.3" >&2
        exit 1
        ;;
esac