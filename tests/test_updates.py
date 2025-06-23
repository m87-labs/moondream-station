import pexpect
import shutil
import os
import logging
import time
import requests
import re
from pathlib import Path

GLOBAL_TIMEOUT = 300
MANIFEST_DIR = "./test_manifests"

def setup_logging(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('test_complete_update_suite.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)

def start_server(executable_path='./moondream_station', args=None):
    cmd = [executable_path]
    if args:
        cmd.extend(args)
    logging.debug(f"Starting server: {' '.join(cmd)}")
    time.sleep(2)
    try:
        child = pexpect.spawn(' '.join(cmd))
        child.expect('moondream>', timeout=100)
        logging.debug("Server started successfully")
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
        if child.isalive():
            child.close(force=True)

def update_manifest_version(version):
    manifest_file = Path(MANIFEST_DIR) / "manifest.json"
    version_file = Path(MANIFEST_DIR) / f"manifest_v{version:03d}.json"
    if not version_file.exists():
        raise FileNotFoundError(f"Version manifest {version_file} not found")
    shutil.copy2(version_file, manifest_file)
    logging.debug(f"Updated manifest.json to version {version:03d}")

def run_admin_command(child, command, expect_pattern=None, timeout=100, expect_exit=False):
    logging.debug(f"Running: {command}")
    child.sendline(command)
    if expect_exit:
        try:
            if expect_pattern:
                child.expect(expect_pattern, timeout=timeout)
                logging.debug(f"Found expected pattern: {expect_pattern}")
            time.sleep(3)
            output = child.before.decode().strip() if hasattr(child, 'before') else ""
            logging.debug("Server exited as expected after update")
            return output
        except pexpect.EOF:
            output = child.before.decode().strip() if hasattr(child, 'before') else ""
            logging.debug("Server exited as expected after update (EOF)")
            return output
        except pexpect.TIMEOUT:
            logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
            output = child.before.decode().strip() if hasattr(child, 'before') else ""
            return output
    else:
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
    run_admin_command(child, 'admin update-manifest', timeout=100, expect_exit=False)

def parse_model_list(output):
    models = {}
    current_model = None
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('Model: '):
            current_model = line[7:].strip()
            models[current_model] = {}
        elif current_model and line.startswith('Release Date: '):
            models[current_model]['release_date'] = line[14:].strip()
        elif current_model and line.startswith('Size: '):
            models[current_model]['model_size'] = line[6:].strip()
        elif current_model and line.startswith('Notes: '):
            models[current_model]['notes'] = line[7:].strip()
    return models

def validate_model_list(model_list_output):
    manifest_path = os.path.expanduser("~/.local/share/MoondreamStation/manifest.py")
    if not os.path.exists(manifest_path):
        logging.warning("manifest.py not found - skipping model list validation")
        return True
    
    with open(manifest_path, 'r') as f:
        manifest_content = f.read()
    
    url_match = re.search(r'MANIFEST_URL\s*=\s*["\']([^"\']+)["\']', manifest_content)
    if not url_match:
        logging.warning("MANIFEST_URL not found in manifest.py - skipping validation")
        return True
    
    try:
        response = requests.get(url_match.group(1), timeout=100)
        manifest_data = response.json()
        logging.debug(f"Fetched manifest from: {url_match.group(1)}")
    except Exception as e:
        logging.warning(f"Failed to fetch manifest: {e}")
        return True
    
    cli_models = parse_model_list(model_list_output)
    manifest_models = manifest_data.get('models', {}).get('2b', {})
    logging.debug("--- Model List Validation ---")
    
    all_valid = True
    for model_name, cli_data in cli_models.items():
        if model_name in manifest_models:
            manifest_model = manifest_models[model_name]
            matches = {
                'release_date': cli_data.get('release_date') == manifest_model.get('release_date'),
                'model_size': cli_data.get('model_size') == manifest_model.get('model_size'),
                'notes': cli_data.get('notes') == manifest_model.get('notes')
            }
            model_valid = all(matches.values())
            all_valid = all_valid and model_valid
            logging.debug(f"Model '{model_name}': {'PASS' if model_valid else 'FAIL'}")
            if not model_valid:
                for field, match in matches.items():
                    if not match:
                        logging.debug(f"  {field}: expected '{manifest_model.get(field)}', got '{cli_data.get(field)}'")
        else:
            logging.warning(f"Model '{model_name}' found in CLI but not in manifest")
            all_valid = False
    
    for model_name in manifest_models:
        if model_name not in cli_models:
            logging.warning(f"Model '{model_name}' in manifest but not in CLI output")
            all_valid = False
    
    logging.debug(f"Model list validation: {'PASS' if all_valid else 'FAIL'}")
    return all_valid

def execute_full_update_command(child, command, executable_path, args, update_type="general"):
    logging.debug(f"Executing full update ({update_type}): {command}")
    try:
        child.sendline(command)
        try:
            if update_type == "model":
                completion_pattern = r'All component updates have been processed'
            elif update_type == "cli":
                completion_pattern = r'CLI update complete\. Please restart the CLI'
            else:
                completion_pattern = r'(All component updates have been processed|CLI update complete\. Please restart)'
            
            index = child.expect([completion_pattern, r'moondream>'], timeout=300)
            
            if index == 0:
                logging.debug(f"Found completion message for {update_type} update - manually exiting as expected")
                child.sendline('exit')
                try:
                    child.expect(r'Exiting Moondream CLI', timeout=5)
                    logging.debug(f"Successfully exited CLI after {update_type} update")
                except pexpect.TIMEOUT:
                    logging.debug(f"CLI exit message not received (expected during {update_type} update)")
                except pexpect.EOF:
                    logging.debug(f"CLI process ended during {update_type} update (expected)")
            else:
                logging.warning(f"Full update ({update_type}) returned to prompt unexpectedly")
                
        except pexpect.TIMEOUT:
            logging.warning(f"Timeout waiting for {update_type} update completion - manually exiting")
            child.sendline('exit')
            try:
                child.expect(r'Exiting Moondream CLI', timeout=5)
            except pexpect.TIMEOUT:
                pass
        
        if child.isalive():
            child.close(force=True)
        
        try:
            child = start_server(executable_path, args)
            logging.debug(f"Server restarted successfully after {update_type} update")
            return child, True
        except Exception as restart_error:
            logging.error(f"Failed to restart server after {update_type} update: {restart_error}")
            return None, False
        
    except Exception as e:
        logging.error(f"Full update ({update_type}) command failed: {e}")
        try:
            if child.isalive():
                child.close(force=True)
            child = start_server(executable_path, args)
            logging.debug(f"Server recovered after {update_type} update failure")
            return child, False
        except:
            logging.error("Failed to recover server")
            return None, False

def test_model_list_and_validation(child):
    logging.debug("=== Testing Model List ===")
    output = run_admin_command(child, 'admin model-list', expect_exit=False)
    logging.debug("Model list output:")
    logging.debug(output)
    success = validate_model_list(output)
    if success:
        logging.debug("Model list validation passed")
    else:
        logging.error("Model list validation failed")
    return success

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
            logging.debug(f"{component}: {actual_status}")
        else:
            logging.error(f"{component}: got '{actual_status}', expected '{expected_status}'")
            success = False
    return success

def execute_bootstrap_update_command(child, command, executable_path, args):
    logging.debug(f"Executing bootstrap update: {command}")
    try:
        output = run_admin_command(child, command, 
                                 expect_pattern=r'(Restart.*for update|Starting update process)',
                                 timeout=120, expect_exit=True)
        logging.debug("Bootstrap update command completed successfully")
        if child.isalive():
            child.close(force=True)
        try:
            child = start_server(executable_path, args)
            logging.debug("Server restarted successfully after bootstrap update")
            return child, True
        except Exception as restart_error:
            logging.error(f"Failed to restart server after bootstrap update: {restart_error}")
            time.sleep(5)
            try:
                child = start_server(executable_path, args)
                logging.debug("Server restarted on second attempt")
                return child, True
            except Exception as second_error:
                logging.error(f"Failed to restart server on second attempt: {second_error}")
                return None, False
    except Exception as e:
        logging.error(f"Bootstrap update command failed: {e}")
        try:
            if child.isalive():
                child.close(force=True)
            time.sleep(3)
            child = start_server(executable_path, args)
            logging.debug("Server recovered after bootstrap update failure")
            return child, False
        except:
            logging.error("Failed to recover server")
            return None, False

def execute_hypervisor_update_command(child, command, executable_path, args):
    logging.debug(f"Executing hypervisor update: {command}")
    try:
        child.sendline(command)
        try:
            index = child.expect([
                r'Hypervisor.*update.*completed',
                r'Server status: Hypervisor: off, Inference: off',
                r'moondream>'
            ], timeout=120)
            
            if index == 0:
                logging.debug("Hypervisor update completed successfully")
                child.expect('moondream>', timeout=10)
            elif index == 1:
                logging.debug("Found 'Hypervisor: off, Inference: off' state - manually exiting as expected")
                child.sendline('exit')
                try:
                    child.expect(r'Exiting Moondream CLI', timeout=5)
                    logging.debug("Successfully exited CLI after hypervisor update")
                except pexpect.TIMEOUT:
                    logging.debug("CLI exit message not received (expected during hypervisor update)")
                except pexpect.EOF:
                    logging.debug("CLI process ended during hypervisor update (expected)")
            else:
                logging.warning("Hypervisor update returned to prompt unexpectedly")
        except pexpect.TIMEOUT:
            logging.warning("Timeout waiting for hypervisor update messages - manually exiting")
            child.sendline('exit')
            try:
                child.expect(r'Exiting Moondream CLI', timeout=10)
            except pexpect.TIMEOUT:
                pass
        
        if child.isalive():
            child.close(force=True)
        
        try:
            child = start_server(executable_path, args)
            logging.debug("Server restarted successfully after hypervisor update")
            return child, True
        except Exception as restart_error:
            logging.error(f"Failed to restart server after hypervisor update: {restart_error}")
            return None, False
    except Exception as e:
        logging.error(f"Hypervisor update command failed: {e}")
        try:
            if child.isalive():
                child.close(force=True)
            child = start_server(executable_path, args)
            logging.debug("Server recovered after hypervisor update failure")
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
    logging.debug("Test environment verified")

def test_incremental_updates_suite(executable_path='./moondream_station', args=None, cleanup=True):
    logging.debug("Starting bootstrap, hypervisor, model, and CLI update test...")
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
        
        child, update_success = execute_bootstrap_update_command(child, 'admin update-bootstrap --confirm', executable_path, args)
        if not update_success or child is None:
            logging.error("Bootstrap update failed or server couldn't restart")
            all_passed = False
            if child is None:
                logging.error("Aborting tests - no server available")
                return False
        
        # Step 3: Hypervisor + Model updates available
        update_manifest_version(3)
        update_server_manifest(child)
        success = check_updates_and_verify(child, "Hypervisor + Model Updates Available (v003)", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Update available',
            'CLI': 'Up to date',
            'Model': 'Update available'
        })
        if not success:
            all_passed = False
        
        # Execute hypervisor update first
        child, update_success = execute_hypervisor_update_command(child, 'admin update-hypervisor --confirm', executable_path, args)
        if not update_success or child is None:
            logging.error("Hypervisor update failed or server couldn't restart")
            all_passed = False
            if child is None:
                logging.error("Aborting tests - no server available")
                return False
        
        # Verify hypervisor updated, model still pending
        success = check_updates_and_verify(child, "After Hypervisor Update in v003", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Up to date',
            'CLI': 'Up to date',
            'Model': 'Update available'
        })
        if not success:
            all_passed = False
        
        # Execute model update
        child, update_success = execute_full_update_command(child, 'admin update --confirm', executable_path, args, "model")
        if not update_success:
            logging.error("Model update (full update) failed")
            all_passed = False
        
        # Verify model updated
        success = check_updates_and_verify(child, "After Model Update in v003", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Up to date',
            'CLI': 'Up to date',
            'Model': 'Up to date'
        })
        if not success:
            all_passed = False
        
        # Validate model list against manifest
        model_list_success = test_model_list_and_validation(child)
        if not model_list_success:
            all_passed = False
        
        # Step 4: CLI update available and execute
        update_manifest_version(4)
        update_server_manifest(child)
        success = check_updates_and_verify(child, "CLI Update Available (v004)", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Up to date',
            'CLI': 'Update available',
            'Model': 'Up to date'
        })
        if not success:
            all_passed = False
        
        # Execute CLI update
        child, update_success = execute_full_update_command(child, 'admin update --confirm', executable_path, args, "cli")
        if not update_success or child is None:
            logging.error("CLI update failed or server couldn't restart")
            all_passed = False
            if child is None:
                logging.error("Aborting tests - no server available")
                return False
        
        # Verify CLI updated
        success = check_updates_and_verify(child, "After CLI Update (v004)", {
            'Bootstrap': 'Up to date',
            'Hypervisor': 'Up to date',
            'CLI': 'Up to date',
            'Model': 'Up to date'
        })
        if not success:
            all_passed = False
        
        if all_passed:
            logging.debug("All update tests passed!")
        else:
            logging.error("Some update tests failed!")
        
        return all_passed
        
    finally:
        if cleanup:
            try:
                update_manifest_version(1)
                logging.debug("Manifest restored to v001")
            except Exception as e:
                logging.warning(f"Failed to restore manifest: {e}")
        try:
            if child and child.isalive():
                end_server(child)
        except:
            pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Moondream Station complete update suite: bootstrap, hypervisor, model, and CLI updates')
    parser.add_argument('--executable', default='./moondream_station', help='Path to moondream_station executable')
    parser.add_argument('--verbose', action='store_true', help='Print log messages to console')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip manifest cleanup')
    args, server_args = parser.parse_known_args()
    setup_logging(verbose=args.verbose)
    success = test_incremental_updates_suite(args.executable, server_args, cleanup=not args.no_cleanup)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()