#!/bin/bash

# Detect the operating system
OS=$(uname -s)

# Set download URLs for Mac and Ubuntu
DOWNLOAD_URL_MAC="https://depot.moondream.ai/station/md_station.tar.gz"
DOWNLOAD_URL_UBUNTU="https://depot.moondream.ai/station/md_station_ubuntu.tar.gz"

if [ "$OS" = "Linux" ]; then
    echo "Ubuntu detected. Installing Moondream Station for Ubuntu..."
    DOWNLOAD_URL="$DOWNLOAD_URL_UBUNTU"
    DOWNLOAD_PATH="./md_station_ubuntu.tar.gz"
    
    # Remove existing app data directory if it exists
    MOONDREAM_DATA_DIR="$HOME/.local/share/MoondreamStation"
    if [ -d "$MOONDREAM_DATA_DIR" ]; then
        echo "Removing existing app data directory..."
        rm -rf "$MOONDREAM_DATA_DIR"
    fi
    
    echo "Downloading Moondream Station..."
    curl -# -L "$DOWNLOAD_URL" -o "$DOWNLOAD_PATH"
    
    if [ $? -ne 0 ]; then
        echo "Download failed. Please check your internet connection and try again."
        exit 1
    fi
    
    echo "Download complete: $DOWNLOAD_PATH"
    
    echo "Extracting files..."
    tar -xzf "$DOWNLOAD_PATH"
    
    if [ $? -ne 0 ]; then
        echo "Extraction failed."
        exit 1
    fi
    
    # Clean up
    echo "Cleaning up temporary files..."
    rm -f "$DOWNLOAD_PATH"
    
    echo "Moondream Station has been successfully installed to the current directory."
    echo "Installation complete! Run Moondream Station to launch the app."
    exit 0
else
    # Mac OS X installation path
    DOWNLOAD_URL="$DOWNLOAD_URL_MAC"
    DOWNLOAD_DIR="$HOME/Downloads"
    DOWNLOAD_PATH="$DOWNLOAD_DIR/moondream_station.tar.gz"
    APPLICATIONS_DIR="/Applications"
    FINAL_DOWNLOAD_DIR="$DOWNLOAD_DIR/Moondream Station"
    
    # Remove existing app data directory if it exists
    MOONDREAM_DATA_DIR="$HOME/Library/MoondreamStation"
    if [ -d "$MOONDREAM_DATA_DIR" ]; then
        echo "Removing existing app data directory..."
        rm -rf "$MOONDREAM_DATA_DIR"
    fi
    
    echo "Downloading Moondream Station..."
    curl -# -L "$DOWNLOAD_URL" -o "$DOWNLOAD_PATH"

if [ $? -ne 0 ]; then
    echo "Download failed. Please check your internet connection and try again."
    exit 1
fi

echo "Download complete: $DOWNLOAD_PATH"

TEMP_EXTRACT_DIR="$DOWNLOAD_DIR/moondream_station_temp"
mkdir -p "$TEMP_EXTRACT_DIR"

echo "Extracting files..."
tar -xzf "$DOWNLOAD_PATH" -C "$TEMP_EXTRACT_DIR"

if [ $? -ne 0 ]; then
    echo "Extraction failed."
    rm -rf "$TEMP_EXTRACT_DIR"
    exit 1
fi

APP_DIR=$(find "$TEMP_EXTRACT_DIR" -name "*.app" -type d -maxdepth 2 | head -1)

if [ -z "$APP_DIR" ]; then
    echo "Could not find the Moondream Station application in the extracted files."
    rm -rf "$TEMP_EXTRACT_DIR"
    exit 1
fi

echo "Attempting to install to Applications folder..."

# Check if we have permission to write to Applications
if [ -w "$APPLICATIONS_DIR" ]; then
    # Remove existing app if it exists
    if [ -d "$APPLICATIONS_DIR/Moondream Station.app" ]; then
        echo "Removing previous installation..."
        rm -rf "$APPLICATIONS_DIR/Moondream Station.app"
    fi
    
    # Copy the app to Applications
    cp -R "$APP_DIR" "$APPLICATIONS_DIR/"
    
    if [ $? -ne 0 ]; then
        echo "Failed to copy to Applications folder, falling back to Downloads folder."
        INSTALL_LOCATION="downloads"
    else
        INSTALL_PATH="$APPLICATIONS_DIR/$(basename "$APP_DIR")"
        INSTALL_LOCATION="applications"
        INSTALL_DIR="$APPLICATIONS_DIR"
    fi
else
    # If we don't have permission, ask for sudo
    echo "Administrator privileges required to install to Applications folder."
    echo "Attempting to install with administrator privileges..."
    
    # Try with sudo, but don't exit on failure
    sudo cp -R "$APP_DIR" "$APPLICATIONS_DIR/"
    
    if [ $? -ne 0 ]; then
        echo "Could not install to Applications folder, falling back to Downloads folder."
        INSTALL_LOCATION="downloads"
    else
        INSTALL_PATH="$APPLICATIONS_DIR/$(basename "$APP_DIR")"
        INSTALL_LOCATION="applications"
        INSTALL_DIR="$APPLICATIONS_DIR"
    fi
fi

# If we couldn't install to Applications, install to Downloads
if [ "$INSTALL_LOCATION" = "downloads" ]; then
    # Prepare Downloads destination
    mkdir -p "$FINAL_DOWNLOAD_DIR"
    
    # Remove existing app if it exists in Downloads
    if [ -d "$FINAL_DOWNLOAD_DIR/$(basename "$APP_DIR")" ]; then
        echo "Removing previous installation from Downloads..."
        rm -rf "$FINAL_DOWNLOAD_DIR/$(basename "$APP_DIR")"
    fi
    
    # Copy to Downloads folder
    echo "Installing to Downloads folder instead..."
    cp -R "$APP_DIR" "$FINAL_DOWNLOAD_DIR/"
    
    if [ $? -ne 0 ]; then
        echo "Failed to copy to Downloads folder."
        rm -rf "$TEMP_EXTRACT_DIR"
        exit 1
    fi
    
    INSTALL_PATH="$FINAL_DOWNLOAD_DIR/$(basename "$APP_DIR")"
    INSTALL_DIR="$FINAL_DOWNLOAD_DIR"
fi

# Clean up
echo "Cleaning up temporary files..."
rm -f "$DOWNLOAD_PATH"
rm -rf "$TEMP_EXTRACT_DIR"

echo "Moondream Station has been successfully installed to: $INSTALL_PATH"

# Open the directory containing the application
echo "Opening the directory containing Moondream Station..."
open "$INSTALL_DIR"

echo "Installation complete! Double-click Moondream Station to launch the app."
fi