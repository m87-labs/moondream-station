"""
System-aware PyTorch installation utility.
Detects system capabilities and installs the appropriate PyTorch version.
"""

import subprocess
import sys
import platform
import logging
from typing import Tuple, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TorchInstaller:
    """Handles system-aware PyTorch installation"""
    
    def __init__(self):
        self.os_name = platform.system().lower()
        self.machine = platform.machine().lower()
        self.is_windows = self.os_name == "windows"
        self.is_linux = self.os_name == "linux"
        self.is_mac = self.os_name == "darwin"
        
    def detect_system_capabilities(self) -> dict:
        """Detect system capabilities for PyTorch installation"""
        capabilities = {
            "os": self.os_name,
            "machine": self.machine,
            "cuda_available": False,
            "cuda_version": None,
            "mps_available": False,
            "recommended_install": "cpu"
        }
        
        # Check for NVIDIA GPU and CUDA
        if self._has_nvidia_gpu():
            cuda_version = self._detect_cuda_version()
            if cuda_version:
                capabilities["cuda_available"] = True
                capabilities["cuda_version"] = cuda_version
                capabilities["recommended_install"] = f"cu{cuda_version.replace('.', '')}"
        
        # Check for Apple Metal Performance Shaders (M1/M2 Macs)
        if self.is_mac and self._has_mps():
            capabilities["mps_available"] = True
            capabilities["recommended_install"] = "cpu"  # MPS uses CPU install
            
        return capabilities
    
    def _has_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            # Try nvidia-smi command
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version from nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Get CUDA runtime version
                cuda_result = subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if cuda_result.returncode == 0:
                    output = cuda_result.stdout
                    # Parse CUDA version from nvidia-smi output
                    for line in output.split('\n'):
                        if 'CUDA Version:' in line:
                            cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                            # Map to PyTorch compatible versions
                            return self._map_cuda_to_pytorch_version(cuda_version)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _map_cuda_to_pytorch_version(self, cuda_version: str) -> str:
        """Map CUDA version to PyTorch compatible version"""
        version_parts = cuda_version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        # Map to available PyTorch CUDA versions
        if major >= 12 and minor >= 4:
            return "12.4"
        elif major >= 12 and minor >= 1:
            return "12.1"
        elif major >= 11 and minor >= 8:
            return "11.8"
        else:
            return "11.8"  # Fallback to oldest supported
    
    def _has_mps(self) -> bool:
        """Check if Apple Metal Performance Shaders is available"""
        if not self.is_mac:
            return False
        try:
            # Check if this is Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "Apple" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_torch_install_command(self, torch_version: str = ">=2.7.0") -> Tuple[List[str], Optional[str]]:
        """Generate PyTorch installation command based on system capabilities"""
        capabilities = self.detect_system_capabilities()
        
        base_packages = ["torch", "torchvision", "torchaudio"]
        
        # Extract version constraint
        if torch_version.startswith(">="):
            version_spec = torch_version[2:]
        elif torch_version.startswith("=="):
            version_spec = torch_version[2:]
        else:
            version_spec = torch_version
            
        # Build install command
        cmd = [sys.executable, "-m", "pip", "install"]
        index_url = None
        
        if capabilities["cuda_available"]:
            # CUDA installation
            cuda_version = capabilities["cuda_version"]
            index_url = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
            
            logger.info(f"Installing PyTorch with CUDA {cuda_version} support")
            
        elif capabilities["mps_available"]:
            # MPS (Apple Silicon) - use default PyPI
            logger.info("Installing PyTorch with MPS (Metal) support for Apple Silicon")
            
        else:
            # CPU-only installation
            index_url = "https://download.pytorch.org/whl/cpu"
            logger.info("Installing CPU-only PyTorch")
        
        # Add packages with version constraints
        for package in base_packages:
            if package == "torch":
                cmd.append(f"{package}{torch_version}")
            else:
                cmd.append(package)
        
        if index_url:
            cmd.extend(["--index-url", index_url])
            
        return cmd, index_url
    
    def install_torch_requirements(self, requirements_content: str) -> bool:
        """Install PyTorch requirements with system-aware selection"""
        try:
            # Parse requirements to find torch
            lines = requirements_content.strip().split('\n')
            torch_line = None
            other_requirements = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                package_name = line.split('>=')[0].split('==')[0].split('<')[0].strip()
                if package_name.lower() == 'torch':
                    torch_line = line
                else:
                    other_requirements.append(line)
            
            # Install PyTorch with system detection
            if torch_line:
                version_constraint = ">=2.7.0"  # Default
                if '>=' in torch_line:
                    version_constraint = '>=' + torch_line.split('>=')[1].strip()
                elif '==' in torch_line:
                    version_constraint = '==' + torch_line.split('==')[1].strip()
                    
                cmd, index_url = self.get_torch_install_command(version_constraint)
                
                logger.info(f"Installing PyTorch with command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"PyTorch installation failed: {result.stderr}")
                    return False
            
            # Install other requirements normally
            if other_requirements:
                temp_requirements = "\n".join(other_requirements)
                temp_file = Path.cwd() / "temp_requirements.txt"
                
                with open(temp_file, "w") as f:
                    f.write(temp_requirements)
                
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(temp_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                temp_file.unlink(missing_ok=True)
                
                if result.returncode != 0:
                    logger.error(f"Other requirements installation failed: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error installing torch requirements: {e}")
            return False
    
    def verify_torch_installation(self) -> dict:
        """Verify PyTorch installation and capabilities"""
        try:
            import torch
            
            info = {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            }
            
            if info["cuda_available"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
            
            return info
            
        except ImportError:
            return {"error": "PyTorch not installed"}
        except Exception as e:
            return {"error": str(e)}


def get_system_torch_installer() -> TorchInstaller:
    """Factory function to get a TorchInstaller instance"""
    return TorchInstaller()


# Utility functions for backward compatibility
def detect_system_torch_requirements() -> dict:
    """Detect system requirements for PyTorch installation"""
    installer = TorchInstaller()
    return installer.detect_system_capabilities()


def get_system_aware_torch_command(torch_version: str = ">=2.7.0") -> Tuple[List[str], Optional[str]]:
    """Get system-aware PyTorch installation command"""
    installer = TorchInstaller()
    return installer.get_torch_install_command(torch_version)
