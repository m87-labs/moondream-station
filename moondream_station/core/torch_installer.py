"""
System-aware PyTorch installation utility.
Detects system capabilities and installs the appropriate PyTorch version.
"""

import subprocess
import sys
import platform
import logging
import json
import hashlib
import tempfile
import contextlib
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

logger = logging.getLogger(__name__)


class TorchInstallResult:
    """Result object for torch installation operations"""
    def __init__(self, success: bool, message: str = "", details: Dict[str, Any] = None, cmd: List[str] = None):
        self.success = success
        self.message = message
        self.details = details or {}
        self.cmd = cmd or []


class TorchInstaller:
    """Handles system-aware PyTorch installation"""
    
    def __init__(self):
        self.os_name = platform.system().lower()
        self.machine = platform.machine().lower()
        self.is_windows = self.os_name == "windows"
        self.is_linux = self.os_name == "linux"
        self.is_mac = self.os_name == "darwin"
        self._capability_cache = None
        self._cache_file = Path.home() / ".moondream-station" / "cache" / "capabilities.json"
        
    def detect_system_capabilities(self) -> dict:
        """Detect system capabilities for PyTorch installation with caching"""
        # Check cache first
        if self._capability_cache:
            return self._capability_cache
            
        cached_data = self._load_capability_cache()
        if cached_data:
            self._capability_cache = cached_data
            return cached_data
            
        capabilities = {
            "os": self.os_name,
            "machine": self.machine,
            "cuda_available": False,
            "cuda_version": None,
            "mps_available": False,
            "recommended_install": "cpu",
            "detection_errors": []
        }
        
        # Single-call NVIDIA detection for efficiency
        nvidia_info = self._detect_nvidia_info()
        if nvidia_info["success"]:
            capabilities["cuda_available"] = True
            capabilities["cuda_version"] = nvidia_info["cuda_version"]
            capabilities["recommended_install"] = f"cu{nvidia_info['cuda_version'].replace('.', '')}"
            capabilities["gpu_name"] = nvidia_info.get("gpu_name")
        else:
            capabilities["detection_errors"].append(nvidia_info["error"])
        
        # Check for Apple Metal Performance Shaders (M1/M2 Macs)
        if self.is_mac:
            mps_info = self._detect_mps_info()
            if mps_info["success"]:
                capabilities["mps_available"] = True
                capabilities["recommended_install"] = "cpu"  # MPS uses CPU install
                capabilities["mps_os_version"] = mps_info.get("os_version")
            else:
                capabilities["detection_errors"].append(mps_info["error"])
        
        # Cache the results
        self._save_capability_cache(capabilities)
        self._capability_cache = capabilities
        return capabilities
    
    def _load_capability_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached capability detection results"""
        try:
            if not self._cache_file.exists():
                return None
                
            with open(self._cache_file) as f:
                cache_data = json.load(f)
            
            # Check cache age (5 minutes TTL)
            import time
            cache_age = time.time() - cache_data.get("timestamp", 0)
            if cache_age > 300:  # 5 minutes
                return None
                
            return cache_data.get("capabilities")
        except Exception:
            return None
    
    def _save_capability_cache(self, capabilities: Dict[str, Any]):
        """Save capability detection results to cache"""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            import time
            cache_data = {
                "timestamp": time.time(),
                "capabilities": capabilities
            }
            
            with open(self._cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass  # Silently fail on cache save errors
    
    def _detect_nvidia_info(self) -> Dict[str, Any]:
        """Single-call NVIDIA detection for efficiency and reliability"""
        try:
            # Single nvidia-smi call to get all needed info
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode != 0:
                return {"success": False, "error": f"nvidia-smi failed: {result.stderr.strip()}"}
            
            gpu_info = result.stdout.strip()
            if not gpu_info:
                return {"success": False, "error": "No GPUs detected by nvidia-smi"}
            
            # Parse first GPU info
            lines = gpu_info.split('\n')
            if lines:
                gpu_name, driver_version = lines[0].split(', ', 1)
            
            # Get CUDA runtime version from nvidia-smi header
            smi_result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            cuda_version = None
            if smi_result.returncode == 0:
                for line in smi_result.stdout.split('\n'):
                    if 'CUDA Version:' in line:
                        cuda_raw = line.split('CUDA Version:')[1].strip().split()[0]
                        cuda_version = self._map_cuda_to_pytorch_version(cuda_raw)
                        break
            
            if not cuda_version:
                return {"success": False, "error": "Could not parse CUDA version from nvidia-smi"}
            
            return {
                "success": True,
                "cuda_version": cuda_version,
                "gpu_name": gpu_name.strip(),
                "driver_version": driver_version.strip()
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "nvidia-smi timeout - driver may be unresponsive"}
        except FileNotFoundError:
            return {"success": False, "error": "nvidia-smi not found - install NVIDIA drivers or set TORCH_CUDA=cpu"}
        except Exception as e:
            return {"success": False, "error": f"NVIDIA detection failed: {str(e)}"}
    
    def _map_cuda_to_pytorch_version(self, cuda_version: str) -> str:
        """Map CUDA version to PyTorch compatible version with better coverage"""
        try:
            version_parts = cuda_version.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            # Map to available PyTorch CUDA versions (updated for broader support)
            if major >= 13:
                return "12.4"  # CUDA 13.0+ uses PyTorch cu124 build (future-proofing)
            elif major == 12 and minor >= 8:
                return "12.4"  # CUDA 12.8+ uses PyTorch cu124 build
            elif major == 12 and minor >= 4:
                return "12.4"  # CUDA 12.4+ uses PyTorch cu124 build
            elif major == 12 and minor >= 1:
                return "12.1"  # CUDA 12.1-12.3 uses PyTorch cu121 build
            elif major == 12:
                return "12.1"  # CUDA 12.0 uses PyTorch cu121 build
            elif major == 11 and minor >= 8:
                return "11.8"  # CUDA 11.8+ uses PyTorch cu118 build
            else:
                logger.warning(f"CUDA {cuda_version} is older than supported versions. Using cu118 as fallback.")
                return "11.8"  # Fallback to oldest supported
        except (ValueError, IndexError):
            logger.warning(f"Invalid CUDA version format: {cuda_version}. Using cu121 as safe fallback.")
            return "12.1"  # Safe fallback
    
    def _detect_mps_info(self) -> Dict[str, Any]:
        """Comprehensive Apple MPS detection with version checking"""
        if not self.is_mac:
            return {"success": False, "error": "Not macOS"}
            
        try:
            # Check if this is Apple Silicon
            cpu_result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if cpu_result.returncode != 0 or "Apple" not in cpu_result.stdout:
                return {"success": False, "error": "Not Apple Silicon - MPS requires M1/M2/M3 chips"}
            
            # Check macOS version (MPS requires 12.3+)
            version_result = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if version_result.returncode == 0:
                version_str = version_result.stdout.strip()
                try:
                    # Parse version (e.g., "13.2.1" -> [13, 2, 1])
                    version_parts = [int(x) for x in version_str.split('.')]
                    if len(version_parts) >= 2:
                        major, minor = version_parts[0], version_parts[1]
                        if major < 12 or (major == 12 and minor < 3):
                            return {
                                "success": False, 
                                "error": f"macOS {version_str} too old for MPS (requires 12.3+)"
                            }
                except ValueError:
                    pass  # Continue with detection if version parsing fails
            
            return {
                "success": True,
                "os_version": version_result.stdout.strip() if version_result.returncode == 0 else "unknown",
                "cpu_info": cpu_result.stdout.strip()
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "System detection timeout"}
        except FileNotFoundError:
            return {"success": False, "error": "System utilities not found"}
        except Exception as e:
            return {"success": False, "error": f"MPS detection failed: {str(e)}"}
    
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
    
    def install_torch_requirements(self, requirements_content: str) -> TorchInstallResult:
        """Install PyTorch requirements with robust parsing and system-aware selection"""
        try:
            # Parse requirements using packaging library for robustness
            parsed_reqs = self._parse_requirements(requirements_content)
            torch_req = None
            other_requirements = []
            
            for line in requirements_content.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                    
                try:
                    req = Requirement(line)
                    if req.name.lower() == 'torch':
                        torch_req = req
                    else:
                        other_requirements.append(line)
                except Exception:
                    # If parsing fails, include in other requirements
                    other_requirements.append(line)
            
            # Install PyTorch with system detection
            if torch_req:
                torch_result = self._install_torch_with_detection(torch_req)
                if not torch_result.success:
                    return torch_result
            
            # Install other requirements normally
            if other_requirements:
                other_result = self._install_other_requirements(other_requirements)
                if not other_result.success:
                    return other_result
            
            # Verify installation
            verification = self.verify_torch_installation()
            if "error" in verification:
                return TorchInstallResult(
                    success=False,
                    message=f"Installation verification failed: {verification['error']}",
                    details=verification
                )
            
            return TorchInstallResult(
                success=True,
                message="PyTorch requirements installed successfully",
                details=verification
            )
            
        except Exception as e:
            logger.error(f"Error installing torch requirements: {e}")
            return TorchInstallResult(
                success=False,
                message=f"Installation failed: {str(e)}",
                details={"exception": str(e)}
            )
    
    def _parse_requirements(self, requirements_content: str) -> List[Requirement]:
        """Parse requirements using packaging library for robustness"""
        requirements = []
        for line in requirements_content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue
            try:
                req = Requirement(line)
                requirements.append(req)
            except Exception as e:
                logger.warning(f"Could not parse requirement line '{line}': {e}")
        return requirements
    
    def _install_torch_with_detection(self, torch_req: Requirement) -> TorchInstallResult:
        """Install PyTorch with system detection and proper error handling"""
        try:
            # Convert requirement to version string
            version_constraint = ">=2.7.0"  # Default
            if torch_req.specifier:
                version_constraint = str(torch_req.specifier)
                
            cmd, index_url = self.get_torch_install_command(version_constraint)
            
            logger.info(f"Installing PyTorch with command: {' '.join(cmd)}")
            if index_url:
                logger.info(f"Using index URL: {index_url}")
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                error_msg = f"PyTorch installation failed (exit code {result.returncode})"
                if "nvidia" in result.stderr.lower() and "not found" in result.stderr.lower():
                    error_msg += ". Try installing with CPU-only: set TORCH_CUDA=cpu"
                elif "timeout" in result.stderr.lower():
                    error_msg += ". Installation timeout - check network connection"
                    
                return TorchInstallResult(
                    success=False,
                    message=error_msg,
                    details={"stderr": result.stderr, "stdout": result.stdout},
                    cmd=cmd
                )
                
            return TorchInstallResult(
                success=True,
                message="PyTorch installed successfully",
                details={"stdout": result.stdout},
                cmd=cmd
            )
            
        except subprocess.TimeoutExpired:
            return TorchInstallResult(
                success=False,
                message="PyTorch installation timeout (>5 minutes)",
                details={"timeout": True},
                cmd=cmd
            )
        except Exception as e:
            return TorchInstallResult(
                success=False,
                message=f"PyTorch installation error: {str(e)}",
                details={"exception": str(e)}
            )
    
    @contextlib.contextmanager
    def _temp_requirements_file(self, requirements: List[str]):
        """Context manager for safe temporary requirements file handling"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(requirements))
            temp_path = f.name
        
        try:
            yield temp_path
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def _install_other_requirements(self, requirements: List[str]) -> TorchInstallResult:
        """Install non-PyTorch requirements with proper error handling"""
        try:
            with self._temp_requirements_file(requirements) as temp_path:
                cmd = [sys.executable, "-m", "pip", "install", "-r", temp_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    return TorchInstallResult(
                        success=False,
                        message=f"Requirements installation failed (exit code {result.returncode})",
                        details={"stderr": result.stderr, "stdout": result.stdout},
                        cmd=cmd
                    )
                    
                return TorchInstallResult(
                    success=True,
                    message="Requirements installed successfully",
                    details={"stdout": result.stdout},
                    cmd=cmd
                )
                
        except subprocess.TimeoutExpired:
            return TorchInstallResult(
                success=False,
                message="Requirements installation timeout",
                details={"timeout": True}
            )
        except Exception as e:
            return TorchInstallResult(
                success=False,
                message=f"Requirements installation error: {str(e)}",
                details={"exception": str(e)}
            )
    
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
