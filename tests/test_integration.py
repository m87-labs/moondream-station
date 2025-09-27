import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from moondream_station.core.torch_installer import TorchInstaller
from moondream_station.core.manifest import ManifestManager


class TestIntegration:
    """Integration tests for the complete installation flow"""
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            "models_dir": str(self.temp_dir / "models")
        }
        self.manifest_manager = ManifestManager(self.config)
    
    def teardown_method(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    @patch('requests.get')
    def test_complete_torch_installation_flow(self, mock_requests, mock_subprocess):
        """Test complete PyTorch installation flow with CUDA detection"""
        # Mock CUDA detection
        mock_subprocess.side_effect = [
            # nvidia-smi query
            Mock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 536.67\n", stderr=""),
            # nvidia-smi version
            Mock(returncode=0, stdout="NVIDIA-SMI 536.67 Driver Version: 536.67 CUDA Version: 12.1\n", stderr=""),
            # pip install torch
            Mock(returncode=0, stdout="Successfully installed torch\n", stderr=""),
            # pip install other requirements
            Mock(returncode=0, stdout="Successfully installed numpy\n", stderr="")
        ]
        
        # Mock requirements content
        requirements_content = """
        torch>=2.7.0
        numpy>=1.21.0
        pillow>=8.0.0
        """
        
        mock_requests.return_value = Mock(
            status_code=200,
            text=requirements_content
        )
        
        installer = TorchInstaller()
        result = installer.install_torch_requirements(requirements_content)
        
        assert result.success is True
        assert "successfully" in result.message
        # Should have called nvidia-smi twice and pip twice
        assert mock_subprocess.call_count == 4
    
    @patch('subprocess.run')
    def test_cpu_fallback_installation(self, mock_subprocess):
        """Test CPU fallback when CUDA not available"""
        # Mock no CUDA available
        mock_subprocess.side_effect = [
            # nvidia-smi not found
            FileNotFoundError(),
            # pip install torch with CPU
            Mock(returncode=0, stdout="Successfully installed torch\n", stderr=""),
            # pip install other requirements
            Mock(returncode=0, stdout="Successfully installed numpy\n", stderr="")
        ]
        
        requirements_content = "torch>=2.7.0\nnumpy>=1.21.0"
        
        installer = TorchInstaller()
        result = installer.install_torch_requirements(requirements_content)
        
        assert result.success is True
        # Verify CPU index URL was used
        torch_cmd = None
        for call in mock_subprocess.call_args_list:
            if 'pip' in str(call) and 'torch' in str(call):
                torch_cmd = call[0][0]
                break
        
        assert torch_cmd is not None
        assert "--index-url" in torch_cmd
        assert "cpu" in torch_cmd[-1]  # CPU index URL
    
    @patch('subprocess.run')
    def test_manifest_torch_installation_integration(self, mock_subprocess):
        """Test PyTorch installation through manifest manager"""
        # Mock successful pip installation
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Successfully installed\n",
            stderr=""
        )
        
        # Create temporary requirements file
        requirements_file = self.temp_dir / "requirements.txt"
        requirements_file.write_text("torch>=2.7.0\nnumpy>=1.21.0")
        
        result = self.manifest_manager._install_requirements(str(requirements_file))
        
        assert result is True
    
    def test_caching_behavior(self):
        """Test that capability detection is cached"""
        installer = TorchInstaller()
        installer._cache_file = self.temp_dir / "capabilities.json"
        
        # Mock capabilities
        capabilities = {
            "os": "linux",
            "cuda_available": True,
            "cuda_version": "12.1",
            "detection_errors": []
        }
        
        # Save to cache
        installer._save_capability_cache(capabilities)
        
        # Create new installer instance
        installer2 = TorchInstaller()
        installer2._cache_file = self.temp_dir / "capabilities.json"
        
        # Should load from cache
        cached_caps = installer2._load_capability_cache()
        assert cached_caps is not None
        assert cached_caps["cuda_available"] is True
        assert cached_caps["cuda_version"] == "12.1"
    
    @patch('subprocess.run')
    def test_error_propagation(self, mock_subprocess):
        """Test that errors are properly propagated with context"""
        # Mock pip failure
        mock_subprocess.side_effect = [
            # nvidia-smi succeeds
            Mock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 536.67\n", stderr=""),
            Mock(returncode=0, stdout="NVIDIA-SMI 536.67 CUDA Version: 12.1\n", stderr=""),
            # pip install fails
            Mock(returncode=1, stdout="", stderr="ERROR: Could not find torch")
        ]
        
        requirements_content = "torch>=2.7.0"
        
        installer = TorchInstaller()
        result = installer.install_torch_requirements(requirements_content)
        
        assert result.success is False
        assert "failed" in result.message
        assert "Could not find torch" in result.details["stderr"]
        assert result.cmd is not None
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_macos_mps_detection_integration(self, mock_subprocess, mock_system):
        """Test macOS MPS detection integration"""
        mock_system.return_value = "Darwin"
        
        installer = TorchInstaller()
        installer.is_mac = True
        
        # Mock Apple Silicon detection
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout="Apple M2 Pro", stderr=""),
            Mock(returncode=0, stdout="13.2.1", stderr=""),
            # pip install with CPU (MPS uses CPU wheel)
            Mock(returncode=0, stdout="Successfully installed torch\n", stderr="")
        ]
        
        capabilities = installer.detect_system_capabilities()
        
        assert capabilities["mps_available"] is True
        assert capabilities["recommended_install"] == "cpu"
    
    def test_requirements_parsing_robustness(self):
        """Test robust requirements parsing with various formats"""
        complex_requirements = """
        # Main PyTorch
        torch>=2.7.0,<3.0.0
        
        # Scientific computing
        numpy>=1.21.0
        scipy>=1.7.0; python_version >= "3.8"
        
        # Vision
        pillow>=8.0.0
        opencv-python==4.8.0.74
        
        # Development
        pytest>=7.0.0
        
        # URLs and extras
        requests[security]==2.28.0
        
        # Comments and empty lines should be ignored
        
        """
        
        installer = TorchInstaller()
        reqs = installer._parse_requirements(complex_requirements)
        
        # Should parse all valid requirements
        req_names = [req.name for req in reqs]
        expected_names = ["torch", "numpy", "scipy", "pillow", "opencv-python", "pytest", "requests"]
        
        for name in expected_names:
            assert name in req_names, f"Expected to find {name} in parsed requirements"
    
    @patch('subprocess.run')
    def test_concurrent_installation_safety(self, mock_subprocess):
        """Test that concurrent installations don't interfere"""
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Successfully installed\n",
            stderr=""
        )
        
        requirements = ["numpy>=1.21.0", "requests==2.28.0"]
        
        installer1 = TorchInstaller()
        installer2 = TorchInstaller()
        
        # Both should be able to create temp files without collision
        with installer1._temp_requirements_file(requirements) as temp1:
            with installer2._temp_requirements_file(requirements) as temp2:
                assert temp1 != temp2  # Different temp files
                assert Path(temp1).exists()
                assert Path(temp2).exists()
        
        # Both should be cleaned up
        assert not Path(temp1).exists()
        assert not Path(temp2).exists()


if __name__ == "__main__":
    pytest.main([__file__])
