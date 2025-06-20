import pexpect
import shutil
import os
import logging
import time
from pathlib import Path

GLOBAL_TIMEOUT = 300
MANIFEST_DIR = "./test_manifests"

def setup_logging(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler('test_bootstrap_update.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

def start_server(executable_path='./moondream_station', args=None):
    cmd = [executable_path]
    if args:
        cmd.extend(args)
    
    logging.debug(f"Starting server: {' '.join(cmd)}")
    
    # Give the system time to clean up any previous processes
    time.sleep(2)
    
    try:
        child = pexpect.spawn(' '.join(cmd))
        child.expect('moondream>', timeout=45)  # Longer timeout for startup
        logging.debug("✓ Server started successfully")
        return child
    except pexpect.EOF:
        logging.error(f"Server failed to start (EOF). Last output: {child.before.decode() if 'child' in locals() else 'None'}")
        raise
    except pexpect.TIMEOUT:
        logging.error(f"Server startup timeout. Output: {child.before.decode() if 'child' in locals() else 'None'}")
        raise
    except Exception as e:
        logging.error(f"Server startup failed: {e}")
        raise

def end_server(child):
    try:
        child.sendline('exit')
        child.expect(r'Exiting Moondream CLI', timeout=10)
        if child.isalive():
            child.close(force=True)
    except:
        # Server might already be dead
        if child.isalive():
            child.close(force=True)

def update_manifest_version(version):
    manifest_file = Path(MANIFEST_DIR) / "manifest.json"
    version_file = Path(MANIFEST_DIR) / f"manifest_v{version:03d}.json"
    
    if not version_file.exists():
        raise FileNotFoundError(f"Version manifest {version_file} not found")
    
    shutil.copy2(version_file, manifest_file)
    logging.debug(f"Updated manifest.json to version {version:03d}")

def run_admin_command(child, command, expect_pattern=None, timeout=60, expect_exit=False):
    logging.debug(f"Running: {command}")
    child.sendline(command)
    
    if expect_exit:
        # For update commands, server will exit - don't wait for prompt
        try:
            if expect_pattern:
                child.expect(expect_pattern, timeout=timeout)
                logging.debug(f"Found expected pattern: {expect_pattern}")
            # Wait a bit more for the server to complete and exit
            time.sleep(3)
            output = child.before.decode().strip() if hasattr(child, 'before') else ""
            logging.debug("Server exited as expected after update")
            return output
        except pexpect.EOF:
            # Expected - server exited after update
            output = child.before.decode().strip() if hasattr(child, 'before') else ""
            logging.debug("Server exited as expected after update (EOF)")
            return output
        except pexpect.TIMEOUT:
            logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
            # Server might have exited anyway
            output = child.before.decode().strip() if hasattr(child, 'before') else ""
            return output
    else:
        # Normal commands - wait for prompt
        if expect_pattern:
            try:
                child.expect(expect_pattern, timeout=timeout)
            except pexpect.TIMEOUT:
                logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
        
        child.expect('moondream>', timeout=timeout)
        output = child.before.decode().strip()
        logging.debug(f"Command output:\n{output}")
        return output

def update_server_manifest(child):
    run_admin_command(child, 'admin update-manifest', timeout=30, expect_exit=False)

def parse_check_updates_output(output):
    components = {}
    
    all_lines = []
    for line in output.replace('\r', '\n').split('\n'):
        line = line.strip()
        if line and not line.startswith('Checking for') and not line.startswith('admin'):
            all_lines.append(line)
    
    for line in all_lines:
        if ':' in line and ('Up to date' in line or 'Update available' in line):
            parts = line.split(':', 1)
            if len(parts) >= 2:
                component = parts[0].strip()
                status_part = parts[1].strip()
                
                component_names = ['Bootstrap', 'Hypervisor', 'CLI', 'Model']
                actual_component = None
                for name in component_names:
                    if name.lower() in component.lower():
                        actual_component = name
                        break
                
                if not actual_component:
                    actual_component = component
                
                if 'Update available' in status_part:
                    components[actual_component] = 'Update available'
                elif 'Up to date' in status_part:
                    components[actual_component] = 'Up to date'
    
    return components

def check_updates_and_verify(child, scenario_name, expected_components):
    logging.debug(f"=== {scenario_name} ===")
    
    output = run_admin_command(child, 'admin check-updates', expect_exit=False)
    components = parse_check_updates_output(output)
    
    success = True
    for component, expected_status in expected_components.items():
        actual_status = components.get(component)
        if actual_status == expected_status:
            logging.debug(f"✓ {component}: {actual_status}")
        else:
            logging.error(f"✗ {component}: got '{actual_status}', expected '{expected_status}'")
            success = False
    
    return success

def execute_update_command(child, command, executable_path, args):
    logging.debug(f"Executing: {command}")
    
    try:
        # Update commands cause server to exit, so expect_exit=True
        output = run_admin_command(child, command, 
                                 expect_pattern=r'(Restart.*for update|Starting update process)',
                                 timeout=120, expect_exit=True)
        logging.debug("✓ Update command completed successfully")
        
        # Close the old child process since server has exited
        if child.isalive():
            child.close(force=True)
        
        # Start a new server instance
        try:
            child = start_server(executable_path, args)
            logging.debug("✓ Server restarted successfully")
            return child, True
        except Exception as restart_error:
            logging.error(f"✗ Failed to restart server after update: {restart_error}")
            # Try one more time
            time.sleep(5)
            try:
                child = start_server(executable_path, args)
                logging.debug("✓ Server restarted on second attempt")
                return child, True
            except Exception as second_error:
                logging.error(f"✗ Failed to restart server on second attempt: {second_error}")
                return None, False
        
    except Exception as e:
        logging.error(f"✗ Update command failed: {e}")
        
        # Try to clean up and start fresh server
        try:
            if child.isalive():
                child.close(force=True)
            time.sleep(3)
            child = start_server(executable_path, args)
            logging.debug("Server recovered after update failure")
            return child, False
        except:
            logging.error("Failed to recover server")
            return None, False

def verify_test_environment():
    manifest_dir = Path(MANIFEST_DIR)
    
    if not manifest_dir.exists():
        raise FileNotFoundError(f"Manifest directory {MANIFEST_DIR} not found")
    
    required_manifests = ['manifest_v001.json', 'manifest_v002.json', 'manifest_v003.json', 'manifest_v004.json']
    missing_manifests = []
    
    for manifest_file in required_manifests:
        if not (manifest_dir / manifest_file).exists():
            missing_manifests.append(manifest_file)
    
    if missing_manifests:
        raise FileNotFoundError(f"Missing manifest files: {missing_manifests}")
    
    logging.debug("✓ Test environment verified")

def test_incremental_updates_suite(executable_path='./moondream_station', args=None, cleanup=True):
    logging.debug("Starting bootstrap update test...")
    
    try:
        verify_test_environment()
    except Exception as e:
        logging.error(f"Test environment verification failed: {e}")
        return False
    
    update_manifest_version(1)
    child = start_server(executable_path, args)
    
    try:
        all_passed = True
        
        # Step 1: Baseline - all up to date
        update_server_manifest(child)
        success = check_updates_and_verify(child, "All Up to Date (v001)", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Up to date', 
            'CLI': 'Up to date'
        })
        if not success:
            all_passed = False
        
        # Step 2: Bootstrap update available and execute
        update_manifest_version(2)
        update_server_manifest(child)
        success = check_updates_and_verify(child, "Bootstrap Update Available (v002)", {
            'Bootstrap': 'Update available',
            'Hypervisor': 'Up to date',
            'CLI': 'Up to date'
        })
        if not success:
            all_passed = False
        
        child, update_success = execute_update_command(child, 'admin update-bootstrap --confirm', executable_path, args)
        if not update_success or child is None:
            logging.error("Bootstrap update failed or server couldn't restart")
            all_passed = False
            if child is None:
                logging.error("Aborting tests - no server available")
                return False
        
        # Step 3: Verify bootstrap is now up to date
        success = check_updates_and_verify(child, "Bootstrap Updated - All Up to Date", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Up to date',
            'CLI': 'Up to date'
        })
        if not success:
            all_passed = False
        
        if all_passed:
            logging.debug("✓ Bootstrap update test passed!")
        else:
            logging.error("✗ Bootstrap update test failed!")
        
        return all_passed
        
    finally:
        if cleanup:
            try:
                update_manifest_version(1)
                logging.debug("✓ Manifest restored to v001")
            except Exception as e:
                logging.warning(f"Failed to restore manifest: {e}")
        
        try:
            if child and child.isalive():
                end_server(child)
        except:
            pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Moondream Station bootstrap updates')
    parser.add_argument('--executable', default='./moondream_station',
                       help='Path to moondream_station executable')
    parser.add_argument('--verbose', action='store_true',
                       help='Print log messages to console')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip manifest cleanup')
    
    args, server_args = parser.parse_known_args()
    
    setup_logging(verbose=args.verbose)
    
    success = test_incremental_updates_suite(args.executable, server_args, cleanup=not args.no_cleanup)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()