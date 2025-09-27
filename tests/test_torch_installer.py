import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from moondream_station.core.torch_installer import TorchInstaller, TorchInstallResult


class TestTorchInstaller:
    def setup_method(self):
        self.installer = TorchInstaller()
    
    def test_cuda_version_mapping(self):
        """Test CUDA version to PyTorch version mapping"""
        test_cases = [
            ("12.4", "12.4"),
            ("12.1", "12.1"),
            ("12.0", "12.1"),
            ("11.8", "11.8"),
            ("11.7", "11.8"),
            ("13.0", "12.4"),  # Future version
            ("10.2", "11.8"),  # Old version fallback
            ("invalid", "12.1"),  # Invalid format fallback
        ]
        
        for cuda_version, expected_pytorch in test_cases:
            result = self.installer._map_cuda_to_pytorch_version(cuda_version)
            assert result == expected_pytorch, f"CUDA {cuda_version} should map to PyTorch cu{expected_pytorch}"
    
    @patch('subprocess.run')
    def test_nvidia_detection_success(self, mock_run):
        """Test successful NVIDIA GPU detection"""
        # Mock nvidia-smi query response
        mock_run.side_effect = [
            Mock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 536.67\n", stderr=""),
            Mock(returncode=0, stdout="NVIDIA-SMI 536.67 Driver Version: 536.67 CUDA Version: 12.2\n", stderr="")
        ]
        
        result = self.installer._detect_nvidia_info()
        
        assert result["success"] is True
        assert result["cuda_version"] == "12.1"
        assert "RTX 4090" in result["gpu_name"]
        assert mock_run.call_count == 2
    
    @patch('subprocess.run')
    def test_nvidia_detection_no_gpu(self, mock_run):
        """Test NVIDIA detection when no GPU present"""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = self.installer._detect_nvidia_info()
        
        assert result["success"] is False
        assert "No GPUs detected" in result["error"]
    
    @patch('subprocess.run')
    def test_nvidia_detection_no_nvidia_smi(self, mock_run):
        """Test NVIDIA detection when nvidia-smi not found"""
        mock_run.side_effect = FileNotFoundError()
        
        result = self.installer._detect_nvidia_info()
        
        assert result["success"] is False
        assert "nvidia-smi not found" in result["error"]
        assert "TORCH_CUDA=cpu" in result["error"]
    
    @patch('subprocess.run')
    def test_nvidia_detection_timeout(self, mock_run):
        """Test NVIDIA detection timeout handling"""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(['nvidia-smi'], 15)
        
        result = self.installer._detect_nvidia_info()
        
        assert result["success"] is False
        assert "timeout" in result["error"]
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_mps_detection_apple_silicon(self, mock_run, mock_system):
        """Test MPS detection on Apple Silicon"""
        mock_system.return_value = "Darwin"
        self.installer.is_mac = True
        
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Apple M2 Pro", stderr=""),
            Mock(returncode=0, stdout="13.2.1", stderr="")
        ]
        
        result = self.installer._detect_mps_info()
        
        assert result["success"] is True
        assert "M2" in result["cpu_info"]
        assert "13.2.1" in result["os_version"]
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_mps_detection_old_macos(self, mock_run, mock_system):
        """Test MPS detection on old macOS version"""
        mock_system.return_value = "Darwin"
        self.installer.is_mac = True
        
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Apple M1", stderr=""),
            Mock(returncode=0, stdout="12.2.0", stderr="")
        ]
        
        result = self.installer._detect_mps_info()
        
        assert result["success"] is False
        assert "too old for MPS" in result["error"]
    
    def test_capability_caching(self):
        """Test capability detection caching"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "capabilities.json"
            self.installer._cache_file = cache_file
            
            # Create mock cache data
            import time
            cache_data = {
                "timestamp": time.time(),
                "capabilities": {
                    "os": "linux",
                    "cuda_available": True,
                    "cuda_version": "12.1"
                }
            }
            
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Should load from cache
            result = self.installer._load_capability_cache()
            assert result is not None
            assert result["cuda_available"] is True
    
    def test_expired_cache(self):
        """Test expired cache handling"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "capabilities.json"
            self.installer._cache_file = cache_file
            
            # Create expired cache data
            import time
            cache_data = {
                "timestamp": time.time() - 400,  # 400 seconds ago (> 5 minutes)
                "capabilities": {"os": "linux"}
            }
            
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
            # Should return None for expired cache
            result = self.installer._load_capability_cache()
            assert result is None
    
    def test_requirements_parsing_with_markers(self):
        """Test requirements parsing with environment markers"""
        requirements_content = """
        torch>=2.7.0
        numpy>=1.21.0
        pillow>=8.0.0; python_version >= "3.8"
        # This is a comment
        requests==2.28.0
        """
        
        reqs = self.installer._parse_requirements(requirements_content)
        
        assert len(reqs) == 4  # torch, numpy, pillow, requests
        torch_req = next((r for r in reqs if r.name == "torch"), None)
        assert torch_req is not None
        assert str(torch_req.specifier) == ">=2.7.0"
    
    def test_requirements_parsing_edge_cases(self):
        """Test requirements parsing with edge cases"""
        requirements_content = """
        torch>=2.7.0,<3.0.0
        git+https://github.com/user/repo.git@main#egg=custom-package
        package[extra]==1.0.0
        -r other_requirements.txt
        --index-url https://pypi.org/simple/
        """
        
        # Should handle complex requirements gracefully
        reqs = self.installer._parse_requirements(requirements_content)
        torch_req = next((r for r in reqs if r.name == "torch"), None)
        assert torch_req is not None
    
    @patch('subprocess.run')
    def test_torch_install_timeout(self, mock_run):
        """Test PyTorch installation timeout handling"""
        import subprocess
        from packaging.requirements import Requirement
        
        mock_run.side_effect = subprocess.TimeoutExpired(['pip'], 300)
        
        torch_req = Requirement("torch>=2.7.0")
        result = self.installer._install_torch_with_detection(torch_req)
        
        assert result.success is False
        assert "timeout" in result.message
    
    @patch('subprocess.run')
    def test_torch_install_network_error(self, mock_run):
        """Test PyTorch installation network error handling"""
        from packaging.requirements import Requirement
        
        mock_run.return_value = Mock(
            returncode=1,
            stderr="Could not find a version that satisfies the requirement torch",
            stdout=""
        )
        
        torch_req = Requirement("torch>=2.7.0")
        result = self.installer._install_torch_with_detection(torch_req)
        
        assert result.success is False
        assert "failed" in result.message
        assert result.details["stderr"] is not None
    
    def test_temp_file_context_manager(self):
        """Test temporary file context manager"""
        requirements = ["numpy>=1.21.0", "requests==2.28.0"]
        
        with self.installer._temp_requirements_file(requirements) as temp_path:
            # File should exist and contain requirements
            assert Path(temp_path).exists()
            with open(temp_path) as f:
                content = f.read()
            assert "numpy>=1.21.0" in content
            assert "requests==2.28.0" in content
        
        # File should be cleaned up
        assert not Path(temp_path).exists()
    
    @patch('builtins.__import__')
    def test_verification_success(self, mock_import):
        """Test PyTorch installation verification"""
        mock_torch = Mock()
        mock_torch.__version__ = "2.7.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        mock_torch.version.cuda = "12.1"
        mock_torch.backends.mps.is_available.return_value = False
        
        def side_effect(name, *args, **kwargs):
            if name == 'torch':
                return mock_torch
            return Mock()
        
        mock_import.side_effect = side_effect
        
        result = self.installer.verify_torch_installation()
        
        assert "error" not in result
        assert result["torch_version"] == "2.7.0"
        assert result["cuda_available"] is True
    
    @patch('builtins.__import__')
    def test_verification_no_torch(self, mock_import):
        """Test verification when PyTorch not installed"""
        def side_effect(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("No module named 'torch'")
            return Mock()
        
        mock_import.side_effect = side_effect
        
        result = self.installer.verify_torch_installation()
        
        assert "error" in result
        assert "not installed" in result["error"]


class TestTorchInstallResult:
    def test_result_creation(self):
        """Test TorchInstallResult creation"""
        result = TorchInstallResult(
            success=True,
            message="Installation successful",
            details={"version": "2.7.0"},
            cmd=["pip", "install", "torch"]
        )
        
        assert result.success is True
        assert result.message == "Installation successful"
        assert result.details["version"] == "2.7.0"
        assert result.cmd == ["pip", "install", "torch"]
    
    def test_result_defaults(self):
        """Test TorchInstallResult with defaults"""
        result = TorchInstallResult(success=False)
        
        assert result.success is False
        assert result.message == ""
        assert result.details == {}
        assert result.cmd == []


if __name__ == "__main__":
    pytest.main([__file__])
