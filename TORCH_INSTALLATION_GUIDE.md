# System-Aware PyTorch Installation Guide

## Overview

Moondream Station now includes a system-aware PyTorch installation mechanism that automatically detects your system capabilities and installs the appropriate PyTorch version (CPU, CUDA, or MPS).

## How It Works

The system detects:
- **Operating System**: Windows, Linux, or macOS
- **GPU Availability**: NVIDIA GPU with CUDA support
- **CUDA Version**: Compatible CUDA runtime version
- **Apple Silicon**: M1/M2 chips with Metal Performance Shaders (MPS)

Based on detection results, it automatically selects:
- **CPU-only** installation for systems without dedicated GPU
- **CUDA-enabled** installation for NVIDIA GPUs with appropriate CUDA version
- **MPS-compatible** installation for Apple Silicon Macs

## Key Features

### Automatic System Detection
```python
from moondream_station.core.torch_installer import TorchInstaller

installer = TorchInstaller()
capabilities = installer.detect_system_capabilities()
print(f"Recommended install: {capabilities['recommended_install']}")
```

### Smart Package Installation
- Detects existing PyTorch installations
- Only installs missing packages
- Uses correct PyTorch index URLs for system-specific builds
- Handles version constraints properly

### CUDA Version Mapping
The installer maps detected CUDA versions to PyTorch-compatible versions:
- CUDA 12.4+ → PyTorch cu124
- CUDA 12.1+ → PyTorch cu121  
- CUDA 11.8+ → PyTorch cu118

## Integration with Backend System

The system integrates with Moondream Station's backend requirements system:

1. **Requirements Detection**: When a backend has `torch` in its requirements.txt
2. **System Analysis**: Analyzes hardware capabilities automatically
3. **Smart Installation**: Installs appropriate PyTorch variant with correct index URL
4. **Verification**: Confirms installation and capabilities

## Example Usage

### Testing System Detection
```bash
python test_torch_detection.py
```

This will show:
- Operating system and architecture
- CUDA availability and version
- Recommended PyTorch installation type
- Generated installation commands
- Current PyTorch installation status

### Sample Output
```
=== PyTorch System Detection Test ===

Detecting system capabilities...
Operating System: windows
Machine Architecture: amd64  
CUDA Available: True
CUDA Version: 11.8
MPS Available (Apple Silicon): False
Recommended Install: cu118

==================================================

Generating PyTorch installation command...
Installation command:
python -m pip install torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Manual Override

If automatic detection doesn't work correctly, you can still manually specify PyTorch installation:

```bash
# Force CPU installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Force CUDA 11.8 installation  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Force CUDA 12.1 installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Architecture Benefits

1. **No More Manual Configuration**: Users don't need to figure out CUDA versions or PyTorch variants
2. **Optimal Performance**: Automatically installs GPU-accelerated versions when available
3. **Backwards Compatible**: Works with existing backend system without changes
4. **Cross-Platform**: Handles Windows, Linux, and macOS differences
5. **Version Flexible**: Supports different PyTorch version constraints from backends

## Future Enhancements

- ROCm support for AMD GPUs
- Intel GPU support (XPU)
- More granular CUDA version detection
- Automatic PyTorch version recommendations based on model requirements
