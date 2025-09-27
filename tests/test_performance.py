import time
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from moondream_station.core.torch_installer import TorchInstaller


def test_capability_detection_caching_performance():
    """Benchmark capability detection with and without caching"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = Path(temp_dir) / "capabilities.json"
        
        # Test without cache (cold start)
        installer = TorchInstaller()
        installer._cache_file = cache_file
        
        with patch('subprocess.run') as mock_run:
            # Mock nvidia-smi responses
            mock_query_response = Mock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 536.67\n", stderr="")
            mock_version_response = Mock(returncode=0, stdout="NVIDIA-SMI 536.67 CUDA Version: 12.1\n", stderr="")

            def slow_run(*args, **kwargs):
                time.sleep(0.1)  # Simulate 100ms nvidia-smi call
                command = args[0] if args else []
                if isinstance(command, list) and any("query-gpu" in str(arg) for arg in command):
                    return mock_query_response
                return mock_version_response

            mock_run.side_effect = slow_run
            
            # First call (cold) - should be slower
            start_time = time.time()
            caps1 = installer.detect_system_capabilities()
            cold_time = time.time() - start_time
            
            assert caps1["cuda_available"] is True
            assert cold_time > 0.1  # Should take at least 100ms due to subprocess mock
        
        # Test with cache (warm start)
        installer2 = TorchInstaller()
        installer2._cache_file = cache_file
        
        # Second call (warm) - should use cache and be much faster
        start_time = time.time()
        caps2 = installer2.detect_system_capabilities()
        warm_time = time.time() - start_time
        
        assert caps2["cuda_available"] is True
        assert warm_time < 0.01  # Should be much faster (< 10ms)
        assert warm_time < cold_time / 5  # At least 5x faster
        
        print(f"\nCold detection: {cold_time:.3f}s")
        print(f"Warm detection: {warm_time:.3f}s")
        print(f"Speedup: {cold_time / warm_time:.1f}x")


def test_requirements_parsing_performance():
    """Benchmark requirements parsing with packaging library vs string splitting"""
    
    # Large requirements file for testing
    large_requirements = """
    torch>=2.7.0,<3.0.0
    numpy>=1.21.0
    scipy>=1.7.0
    matplotlib>=3.5.0
    pandas>=1.3.0
    scikit-learn>=1.0.0
    tensorflow>=2.8.0
    keras>=2.8.0
    pillow>=8.0.0
    opencv-python>=4.5.0
    requests>=2.25.0
    urllib3>=1.26.0
    certifi>=2021.5.25
    charset-normalizer>=2.0.0
    idna>=3.2
    pyyaml>=5.4.0
    jinja2>=3.0.0
    markupsafe>=2.0.0
    click>=8.0.0
    flask>=2.0.0
    werkzeug>=2.0.0
    itsdangerous>=2.0.0
    """
    
    installer = TorchInstaller()
    
    # Benchmark robust parsing with packaging library
    start_time = time.time()
    for _ in range(100):  # Parse 100 times to get measurable time
        reqs = installer._parse_requirements(large_requirements)
    robust_time = time.time() - start_time
    
    # Simple string splitting (old method) for comparison
    def simple_parse(content):
        requirements = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            package_name = line.split('>=')[0].split('==')[0].split('<')[0].strip()
            requirements.append(package_name)
        return requirements
    
    start_time = time.time()
    for _ in range(100):
        simple_reqs = simple_parse(large_requirements)
    simple_time = time.time() - start_time
    
    print(f"Robust parsing (100x): {robust_time:.3f}s")
    print(f"Simple parsing (100x): {simple_time:.3f}s")
    print(f"Overhead: {(robust_time / simple_time - 1) * 100:.1f}%")
    
    # Verify robust parsing is more accurate
    assert len(reqs) > len(simple_reqs)  # Should handle complex specs better
    
    # Acceptable overhead (should be < 5x slower)
    assert robust_time < simple_time * 5


def test_memory_usage_optimization():
    """Test memory usage of improved implementation"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    installer = TorchInstaller()
    
    # Create many capability detection instances to test caching
    capabilities_list = []
    for _ in range(10):
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = [
                Mock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 536.67\n", stderr=""),
                Mock(returncode=0, stdout="NVIDIA-SMI 536.67 CUDA Version: 12.1\n", stderr="")
            ]
            caps = installer.detect_system_capabilities()
            capabilities_list.append(caps)
    
    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory
    
    print(f"Memory growth: {memory_growth / 1024 / 1024:.1f} MB")
    
    # Should not grow memory significantly due to caching
    assert memory_growth < 10 * 1024 * 1024  # Less than 10MB growth


if __name__ == "__main__":
    test_capability_detection_caching_performance()
    test_requirements_parsing_performance()
    test_memory_usage_optimization()
    print("\nAll performance tests passed!")
