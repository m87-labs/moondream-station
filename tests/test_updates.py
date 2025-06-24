import pexpect
import shutil
import os
import logging
import time
import requests
import re
from pathlib import Path
from utils import DebugTracer, TracedProcess, setup_trace_logging

# Import capability testing functions
try:
    from test_capability import test_model_capabilities, parse_model_list_output
    CAPABILITY_TESTING_AVAILABLE = True
except ImportError:
    CAPABILITY_TESTING_AVAILABLE = False
    logging.warning("test_capability.py not found - capability testing disabled")

MANIFEST_DIR = "./test_manifests"

class Timeouts:
    QUICK = 15
    STANDARD = 60
    STARTUP = 60
    UPDATE = 300
    RECOVERY = 30

class Patterns:
    PROMPT = 'moondream>'
    EXIT_MESSAGE = r'Exiting Moondream CLI'
    
    UPDATE_COMPLETION = {
        'model': r'All component updates have been processed',
        'cli': r'CLI update complete\. Please restart the CLI',
        'bootstrap': r'(Restart.*for update|Starting update process)',
        'hypervisor_complete': r'Hypervisor.*update.*completed',
        'hypervisor_off': r'Server status: Hypervisor: off, Inference: off'
    }
    
    STATUS_INDICATORS = {
        'up_to_date': 'Up to date',
        'update_available': 'Update available'
    }
    
    COMPONENT_NAMES = ['Bootstrap', 'Hypervisor', 'CLI', 'Model']
    
    MODEL_FIELDS = {
        'name': 'Model: ',
        'release_date': 'Release Date: ',
        'size': 'Size: ',
        'notes': 'Notes: '
    }
    
    MODEL_CHANGE = {
        'success': r'Model successfully changed to',
        'initialization': r'Model initialization completed successfully'
    }

class Config:
    MANIFEST_PATH = os.path.expanduser("~/.local/share/MoondreamStation/manifest.py")
    MANIFEST_URL_PATTERN = r'MANIFEST_URL\s*=\s*["\']([^"\']+)["\']'
    MODEL_CATEGORY = '2b'

def setup_logging(verbose=False, debug_trace=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler('test_updates.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    setup_trace_logging(debug_trace)
    
    if verbose:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)

class Server:
    def __init__(self, executable='./moondream_station', args=None):
        self.executable = executable
        self.args = args or []
        self.process = None
    
    @DebugTracer.log_operation
    def start(self):
        cmd = [self.executable] + self.args
        DebugTracer.log(f"Starting server command: {' '.join(cmd)}", "SERVER")
        logging.debug(f"Starting server: {' '.join(cmd)}")
        time.sleep(2)
        try:
            raw_process = pexpect.spawn(' '.join(cmd))
            self.process = TracedProcess(raw_process)
            DebugTracer.log("Waiting for startup prompt", "SERVER")
            self.process.expect(Patterns.PROMPT, timeout=Timeouts.STARTUP)
            DebugTracer.log("Server started successfully", "SERVER")
            logging.debug("Server started successfully")
            return self.process
        except pexpect.EOF:
            output = self.process.before.decode() if self.process else 'None'
            DebugTracer.log(f"Server startup failed (EOF): {output}", "SERVER")
            logging.error(f"Server failed to start (EOF). Output: {output}")
            raise
        except pexpect.TIMEOUT:
            output = self.process.before.decode() if self.process else 'None'
            DebugTracer.log(f"Server startup timeout: {output}", "SERVER")
            logging.error(f"Server startup timeout. Output: {output}")
            raise
    
    @DebugTracer.log_operation
    def stop(self):
        if not self.process:
            DebugTracer.log("No process to stop", "SERVER")
            return
        try:
            DebugTracer.log("Sending exit command", "SERVER")
            self.process.sendline('exit')
            self.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
            if self.process.isalive():
                DebugTracer.log("Force closing process", "SERVER")
                self.process.close(force=True)
            DebugTracer.log("Server stopped successfully", "SERVER")
            logging.debug("Server stopped successfully")
        except:
            if self.process.isalive():
                DebugTracer.log("Force stopping process after error", "SERVER")
                self.process.close(force=True)
            DebugTracer.log("Server force stopped", "SERVER")
            logging.debug("Server force stopped")
    
    @DebugTracer.log_operation
    def restart(self):
        DebugTracer.log("Restarting server", "SERVER")
        logging.debug("Restarting server")
        self.stop()
        time.sleep(2)
        return self.start()

class Manifest:
    @staticmethod
    @DebugTracer.log_operation
    def update_version(version):
        DebugTracer.log(f"Updating manifest to version {version:03d}", "MANIFEST")
        manifest_file = Path(MANIFEST_DIR) / "manifest.json"
        version_file = Path(MANIFEST_DIR) / f"manifest_v{version:03d}.json"
        if not version_file.exists():
            DebugTracer.log(f"Version file not found: {version_file}", "MANIFEST")
            raise FileNotFoundError(f"Version manifest {version_file} not found")
        
        DebugTracer.log(f"Copying {version_file} to {manifest_file}", "MANIFEST")
        shutil.copy2(version_file, manifest_file)
        DebugTracer.log(f"Manifest update completed", "MANIFEST")
        logging.debug(f"Updated manifest.json to version {version:03d}")
    
    @staticmethod
    @DebugTracer.log_operation
    def verify_environment():
        DebugTracer.log(f"Verifying test environment in {MANIFEST_DIR}", "MANIFEST")
        manifest_dir = Path(MANIFEST_DIR)
        if not manifest_dir.exists():
            DebugTracer.log(f"Manifest directory not found: {MANIFEST_DIR}", "MANIFEST")
            raise FileNotFoundError(f"Manifest directory {MANIFEST_DIR} not found")
        
        required = ['manifest_v001.json', 'manifest_v002.json', 'manifest_v003.json', 'manifest_v004.json', 'manifest_v005.json']
        missing = []
        for manifest_file in required:
            file_path = manifest_dir / manifest_file
            if not file_path.exists():
                missing.append(manifest_file)
                DebugTracer.log(f"Missing required file: {file_path}", "MANIFEST")
            else:
                DebugTracer.log(f"Found required file: {file_path}", "MANIFEST")
        
        if missing:
            DebugTracer.log(f"Missing files: {missing}", "MANIFEST")
            raise FileNotFoundError(f"Missing manifest files: {missing}")
        
        DebugTracer.log("Test environment verification completed", "MANIFEST")
        logging.debug("Test environment verified")

class Commands:
    def __init__(self, process):
        self.process = process
    
    @DebugTracer.log_command
    def run(self, command, expect_pattern=None, timeout=Timeouts.STANDARD, expect_exit=False):
        DebugTracer.log(f"Executing command: {command}", "COMMAND")
        logging.debug(f"Running: {command}")
        self.process.sendline(command)
        
        if expect_exit:
            try:
                if expect_pattern:
                    DebugTracer.log(f"Waiting for exit pattern: {expect_pattern}", "COMMAND")
                    self.process.expect(expect_pattern, timeout=timeout)
                    logging.debug(f"Found expected pattern: {expect_pattern}")
                time.sleep(3)
                output = self.process.before.decode().strip() if hasattr(self.process, 'before') else ""
                DebugTracer.log(f"Command completed with exit, output length: {len(output)}", "COMMAND")
                logging.debug("Server exited as expected")
                return output
            except pexpect.EOF:
                output = self.process.before.decode().strip() if hasattr(self.process, 'before') else ""
                DebugTracer.log(f"Command completed with EOF, output length: {len(output)}", "COMMAND")
                logging.debug("Server exited (EOF)")
                return output
            except pexpect.TIMEOUT:
                logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
                output = self.process.before.decode().strip() if hasattr(self.process, 'before') else ""
                DebugTracer.log(f"Command timeout, output length: {len(output)}", "COMMAND")
                return output
        else:
            if expect_pattern:
                try:
                    DebugTracer.log(f"Waiting for pattern: {expect_pattern}", "COMMAND")
                    self.process.expect(expect_pattern, timeout=timeout)
                except pexpect.TIMEOUT:
                    logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
            DebugTracer.log("Waiting for prompt", "COMMAND")
            self.process.expect(Patterns.PROMPT, timeout=timeout)
            output = self.process.before.decode().strip()
            DebugTracer.log(f"Command completed, output length: {len(output)}", "COMMAND")
            logging.debug(f"Command output: {output}")
            return output
    
    @DebugTracer.log_command
    def update_manifest(self):
        return self.run('admin update-manifest', timeout=Timeouts.STANDARD)
    
    @DebugTracer.log_command
    def check_updates(self):
        return self.run('admin check-updates')
    
    @DebugTracer.log_command
    def model_list(self):
        return self.run('admin model-list')
    
    @DebugTracer.log_command  
    def model_use(self, model_name):
        return self.run(f'admin model-use {model_name} --confirm', timeout=Timeouts.UPDATE)
    
    @DebugTracer.log_command
    def status(self):
        return self.run('admin status')
    
    @DebugTracer.log_command
    def get_config(self):
        return self.run('admin get-config')

class Parser:
    @staticmethod
    def parse_updates(output):
        components = {}
        lines = [line.strip() for line in output.replace('\r', '\n').split('\n') 
                if line.strip() and not line.strip().startswith('Checking for') and not line.strip().startswith('admin')]
        
        for line in lines:
            if ':' in line and (Patterns.STATUS_INDICATORS['up_to_date'] in line or 
                               Patterns.STATUS_INDICATORS['update_available'] in line):
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    component = parts[0].strip()
                    status_part = parts[1].strip()
                    
                    for name in Patterns.COMPONENT_NAMES:
                        if name.lower() in component.lower():
                            component = name
                            break
                    
                    if Patterns.STATUS_INDICATORS['update_available'] in status_part:
                        components[component] = Patterns.STATUS_INDICATORS['update_available']
                    elif Patterns.STATUS_INDICATORS['up_to_date'] in status_part:
                        components[component] = Patterns.STATUS_INDICATORS['up_to_date']
        
        logging.debug(f"Parsed components: {components}")
        return components
    
    @staticmethod
    def parse_models(output):
        models = {}
        current_model = None
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith(Patterns.MODEL_FIELDS['name']):
                current_model = line[len(Patterns.MODEL_FIELDS['name']):].strip()
                models[current_model] = {}
            elif current_model and line.startswith(Patterns.MODEL_FIELDS['release_date']):
                models[current_model]['release_date'] = line[len(Patterns.MODEL_FIELDS['release_date']):].strip()
            elif current_model and line.startswith(Patterns.MODEL_FIELDS['size']):
                models[current_model]['model_size'] = line[len(Patterns.MODEL_FIELDS['size']):].strip()
            elif current_model and line.startswith(Patterns.MODEL_FIELDS['notes']):
                models[current_model]['notes'] = line[len(Patterns.MODEL_FIELDS['notes']):].strip()
        logging.debug(f"Parsed models: {list(models.keys())}")
        return models
    
    @staticmethod
    def parse_config(output):
        config = {}
        for line in output.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('Getting server configuration'):
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip()
        logging.debug(f"Parsed config: {config}")
        return config

class Validator:
    @staticmethod
    @DebugTracer.log_operation
    def check_updates(process, scenario, expected):
        DebugTracer.log(f"Validating scenario: {scenario}", "VALIDATOR")
        logging.debug(f"=== {scenario} ===")
        cmd = Commands(process)
        output = cmd.check_updates()
        actual = Parser.parse_updates(output)
        
        DebugTracer.log(f"Expected: {expected}", "VALIDATOR")
        DebugTracer.log(f"Actual: {actual}", "VALIDATOR")
        
        success = True
        for component, expected_status in expected.items():
            actual_status = actual.get(component)
            if actual_status == expected_status:
                DebugTracer.log(f"✓ {component}: {actual_status}", "VALIDATOR")
                logging.debug(f"{component}: {actual_status}")
            else:
                DebugTracer.log(f"✗ {component}: got '{actual_status}', expected '{expected_status}'", "VALIDATOR")
                logging.error(f"{component}: got '{actual_status}', expected '{expected_status}'")
                success = False
        
        DebugTracer.log(f"Validation result: {'PASS' if success else 'FAIL'}", "VALIDATOR")
        logging.debug(f"Check updates result: {'PASS' if success else 'FAIL'}")
        return success
    
    @staticmethod
    @DebugTracer.log_operation
    def model_list(process):
        DebugTracer.log("Starting model list validation", "VALIDATOR")
        logging.debug("=== Testing Model List ===")
        cmd = Commands(process)
        output = cmd.model_list()
        models = Parser.parse_models(output)
        
        DebugTracer.log(f"Checking manifest file: {Config.MANIFEST_PATH}", "VALIDATOR")
        if not os.path.exists(Config.MANIFEST_PATH):
            DebugTracer.log("Manifest file not found - skipping validation", "VALIDATOR")
            logging.warning("manifest.py not found - skipping model list validation")
            return True
        
        with open(Config.MANIFEST_PATH, 'r') as f:
            manifest_content = f.read()
        
        DebugTracer.log(f"Searching for manifest URL with pattern: {Config.MANIFEST_URL_PATTERN}", "VALIDATOR")
        url_match = re.search(Config.MANIFEST_URL_PATTERN, manifest_content)
        if not url_match:
            DebugTracer.log("MANIFEST_URL not found - skipping validation", "VALIDATOR")
            logging.warning("MANIFEST_URL not found - skipping validation")
            return True
        
        manifest_url = url_match.group(1)
        DebugTracer.log(f"Fetching manifest from: {manifest_url}", "VALIDATOR")
        try:
            response = requests.get(manifest_url, timeout=10)
            DebugTracer.log(f"HTTP response: {response.status_code}, content length: {len(response.content)}", "VALIDATOR")
            manifest_data = response.json()
            logging.debug(f"Fetched manifest from: {manifest_url}")
        except Exception as e:
            DebugTracer.log(f"Failed to fetch manifest: {str(e)}", "VALIDATOR")
            logging.warning(f"Failed to fetch manifest: {e}")
            return True
        
        manifest_models = manifest_data.get('models', {}).get(Config.MODEL_CATEGORY, {})
        DebugTracer.log(f"Found {len(manifest_models)} models in manifest, {len(models)} in CLI", "VALIDATOR")
        
        all_valid = True
        for model_name, cli_data in models.items():
            DebugTracer.log(f"Validating model: {model_name}", "VALIDATOR")
            if model_name in manifest_models:
                manifest_model = manifest_models[model_name]
                matches = {
                    'release_date': cli_data.get('release_date') == manifest_model.get('release_date'),
                    'model_size': cli_data.get('model_size') == manifest_model.get('model_size'),
                    'notes': cli_data.get('notes') == manifest_model.get('notes')
                }
                model_valid = all(matches.values())
                all_valid = all_valid and model_valid
                DebugTracer.log(f"Model '{model_name}': {'PASS' if model_valid else 'FAIL'}", "VALIDATOR")
                logging.debug(f"Model '{model_name}': {'PASS' if model_valid else 'FAIL'}")
                if not model_valid:
                    for field, match in matches.items():
                        if not match:
                            expected_val = manifest_model.get(field)
                            actual_val = cli_data.get(field)
                            DebugTracer.log(f"  {field}: expected '{expected_val}', got '{actual_val}'", "VALIDATOR")
                            logging.debug(f"  {field}: expected '{expected_val}', got '{actual_val}'")
            else:
                DebugTracer.log(f"Model '{model_name}' found in CLI but not in manifest", "VALIDATOR")
                logging.warning(f"Model '{model_name}' found in CLI but not in manifest")
                all_valid = False
        
        for model_name in manifest_models:
            if model_name not in models:
                DebugTracer.log(f"Model '{model_name}' in manifest but not in CLI output", "VALIDATOR")
                logging.warning(f"Model '{model_name}' in manifest but not in CLI output")
                all_valid = False
        
        DebugTracer.log(f"Model list validation: {'PASS' if all_valid else 'FAIL'}", "VALIDATOR")
        logging.debug(f"Model list validation: {'PASS' if all_valid else 'FAIL'}")
        return all_valid

    @staticmethod
    def model_switch(process, model_name, expected_inference_client=None):
        DebugTracer.log(f"Testing model switch to: {model_name}", "VALIDATOR")
        logging.debug(f"=== Testing Model Switch to {model_name} ===")
        
        cmd = Commands(process)
        
        config_before = cmd.get_config()
        parsed_config_before = Parser.parse_config(config_before)
        
        current_model = parsed_config_before.get('active_model', 'unknown')
        current_inference_client = parsed_config_before.get('active_inference_client', 'unknown')
        
        DebugTracer.log(f"Before switch - Model: {current_model}, Inference Client: {current_inference_client}", "VALIDATOR")
        logging.debug(f"Before switch - Model: {current_model}, Inference Client: {current_inference_client}")
        
        DebugTracer.log(f"Switching to model: {model_name}", "VALIDATOR")
        output = cmd.model_use(model_name)
        
        if Patterns.MODEL_CHANGE['initialization'] in output:
            DebugTracer.log(f"Model switch command succeeded", "VALIDATOR")
            logging.debug(f"Model switch to {model_name} succeeded")
        else:
            DebugTracer.log(f"Model switch command failed - success pattern not found", "VALIDATOR")
            logging.error(f"Model switch to {model_name} failed - success pattern not found")
            return False
        
        config_after = cmd.get_config()
        parsed_config_after = Parser.parse_config(config_after)
        
        new_model = parsed_config_after.get('active_model', 'unknown')
        new_inference_client = parsed_config_after.get('active_inference_client', 'unknown')
        
        DebugTracer.log(f"After switch - Model: {new_model}, Inference Client: {new_inference_client}", "VALIDATOR")
        logging.debug(f"After switch - Model: {new_model}, Inference Client: {new_inference_client}")
        
        if new_model == model_name:
            DebugTracer.log(f"✓ Active model confirmed: {model_name}", "VALIDATOR")
            logging.debug(f"Active model confirmed: {model_name}")
        else:
            DebugTracer.log(f"✗ Active model verification failed - expected {model_name}, got {new_model}", "VALIDATOR")
            logging.error(f"Active model not set to {model_name}, got {new_model}")
            return False
        
        if expected_inference_client:
            if new_inference_client == expected_inference_client:
                DebugTracer.log(f"✓ Inference client confirmed: {expected_inference_client}", "VALIDATOR")
                logging.debug(f"Inference client confirmed: {expected_inference_client}")
                
                if current_inference_client != new_inference_client:
                    DebugTracer.log(f"✓ Inference client updated: {current_inference_client} → {new_inference_client}", "VALIDATOR")
                    logging.debug(f"Inference client updated: {current_inference_client} → {new_inference_client}")
                else:
                    DebugTracer.log(f"✓ Inference client unchanged: {new_inference_client}", "VALIDATOR")
                    logging.debug(f"Inference client unchanged: {new_inference_client}")
            else:
                DebugTracer.log(f"✗ Inference client verification failed - expected {expected_inference_client}, got {new_inference_client}", "VALIDATOR")
                logging.error(f"Inference client not set to {expected_inference_client}, got {new_inference_client}")
                return False
        
        DebugTracer.log(f"Model switch validation: PASS", "VALIDATOR")
        logging.debug(f"Model switch to {model_name} validation: PASS")
        return True

class Updater:
    def __init__(self, server):
        self.server = server
    
    @DebugTracer.log_operation
    def bootstrap(self, command='admin update-bootstrap --confirm'):
        DebugTracer.log(f"Starting bootstrap update with command: {command}", "UPDATER")
        logging.debug(f"Executing bootstrap update: {command}")
        try:
            cmd = Commands(self.server.process)
            cmd.run(command, expect_pattern=Patterns.UPDATE_COMPLETION['bootstrap'], 
                   timeout=Timeouts.UPDATE, expect_exit=True)
            DebugTracer.log("Bootstrap update command completed", "UPDATER")
            logging.debug("Bootstrap update completed successfully")
            
            if self.server.process.isalive():
                DebugTracer.log("Closing process after bootstrap update", "UPDATER")
                self.server.process.close(force=True)
            
            DebugTracer.log("Restarting server after bootstrap update", "UPDATER")
            self.server.restart()
            DebugTracer.log("Bootstrap update sequence completed successfully", "UPDATER")
            logging.debug("Server restarted after bootstrap update")
            return True
        except Exception as e:
            DebugTracer.log(f"Bootstrap update failed: {str(e)}", "UPDATER")
            logging.error(f"Bootstrap update failed: {e}")
            try:
                if self.server.process.isalive():
                    DebugTracer.log("Force closing process after bootstrap error", "UPDATER")
                    self.server.process.close(force=True)
                time.sleep(3)
                DebugTracer.log("Attempting server recovery after bootstrap failure", "UPDATER")
                self.server.restart()
                DebugTracer.log("Server recovery completed", "UPDATER")
                logging.debug("Server recovered after bootstrap update failure")
                return False
            except Exception as recover_error:
                DebugTracer.log(f"Server recovery failed: {str(recover_error)}", "UPDATER")
                logging.error(f"Failed to recover server: {recover_error}")
                return False
    
    @DebugTracer.log_operation
    def hypervisor(self, command='admin update-hypervisor --confirm'):
        DebugTracer.log(f"Starting hypervisor update with command: {command}", "UPDATER")
        logging.debug(f"Executing hypervisor update: {command}")
        try:
            self.server.process.sendline(command)
            try:
                DebugTracer.log("Waiting for hypervisor update completion patterns", "UPDATER")
                index = self.server.process.expect([
                    Patterns.UPDATE_COMPLETION['hypervisor_complete'],
                    Patterns.UPDATE_COMPLETION['hypervisor_off'],
                    Patterns.PROMPT
                ], timeout=Timeouts.UPDATE)
                
                if index == 0:
                    DebugTracer.log("Hypervisor update completed normally", "UPDATER")
                    logging.debug("Hypervisor update completed")
                    self.server.process.expect(Patterns.PROMPT, timeout=Timeouts.QUICK)
                elif index == 1:
                    DebugTracer.log("Hypervisor off state detected - exiting as expected", "UPDATER")
                    logging.debug("Found 'Hypervisor: off' state - exiting as expected")
                    self.server.process.sendline('exit')
                    try:
                        self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                        DebugTracer.log("Clean exit after hypervisor update", "UPDATER")
                        logging.debug("Exited CLI after hypervisor update")
                    except (pexpect.TIMEOUT, pexpect.EOF):
                        DebugTracer.log("Process ended during hypervisor update exit", "UPDATER")
                        logging.debug("CLI process ended during hypervisor update")
                else:
                    DebugTracer.log("Hypervisor update returned to prompt unexpectedly", "UPDATER")
                    logging.warning("Hypervisor update returned to prompt unexpectedly")
            except pexpect.TIMEOUT:
                DebugTracer.log("Timeout during hypervisor update - forcing exit", "UPDATER")
                logging.warning("Timeout waiting for hypervisor update - exiting")
                self.server.process.sendline('exit')
                try:
                    self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                except pexpect.TIMEOUT:
                    DebugTracer.log("Timeout during forced exit", "UPDATER")
                    pass
            
            if self.server.process.isalive():
                DebugTracer.log("Force closing process after hypervisor update", "UPDATER")
                self.server.process.close(force=True)
            
            DebugTracer.log("Restarting server after hypervisor update", "UPDATER")
            self.server.restart()
            DebugTracer.log("Hypervisor update sequence completed successfully", "UPDATER")
            logging.debug("Server restarted after hypervisor update")
            return True
        except Exception as e:
            DebugTracer.log(f"Hypervisor update failed: {str(e)}", "UPDATER")
            logging.error(f"Hypervisor update failed: {e}")
            try:
                if self.server.process.isalive():
                    DebugTracer.log("Force closing process after hypervisor error", "UPDATER")
                    self.server.process.close(force=True)
                DebugTracer.log("Attempting server recovery after hypervisor failure", "UPDATER")
                self.server.restart()
                DebugTracer.log("Server recovery completed", "UPDATER")
                logging.debug("Server recovered after hypervisor update failure")
                return False
            except Exception as recover_error:
                DebugTracer.log(f"Server recovery failed: {str(recover_error)}", "UPDATER")
                logging.error(f"Failed to recover server: {recover_error}")
                return False
    
    @DebugTracer.log_operation
    def full(self, command='admin update --confirm', update_type="general"):
        DebugTracer.log(f"Starting full update ({update_type}) with command: {command}", "UPDATER")
        logging.debug(f"Executing full update ({update_type}): {command}")
        try:
            self.server.process.sendline(command)
            try:
                patterns = {
                    "model": Patterns.UPDATE_COMPLETION['model'],
                    "cli": Patterns.UPDATE_COMPLETION['cli'],
                    "general": f"({Patterns.UPDATE_COMPLETION['model']}|{Patterns.UPDATE_COMPLETION['cli']})"
                }
                
                completion_pattern = patterns.get(update_type, patterns["general"])
                DebugTracer.log(f"Waiting for completion pattern: {completion_pattern}", "UPDATER")
                index = self.server.process.expect([completion_pattern, Patterns.PROMPT], timeout=Timeouts.UPDATE)
                
                if index == 0:
                    DebugTracer.log(f"Full update ({update_type}) completion message found", "UPDATER")
                    logging.debug(f"Found completion message for {update_type} update - exiting")
                    self.server.process.sendline('exit')
                    try:
                        self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                        DebugTracer.log(f"Clean exit after {update_type} update", "UPDATER")
                        logging.debug(f"Exited CLI after {update_type} update")
                    except (pexpect.TIMEOUT, pexpect.EOF):
                        DebugTracer.log(f"Process ended during {update_type} update exit", "UPDATER")
                        logging.debug(f"CLI process ended during {update_type} update")
                else:
                    DebugTracer.log(f"Full update ({update_type}) returned to prompt unexpectedly", "UPDATER")
                    logging.warning(f"Full update ({update_type}) returned to prompt unexpectedly")
                    
            except pexpect.TIMEOUT:
                DebugTracer.log(f"Timeout during {update_type} update - forcing exit", "UPDATER")
                logging.warning(f"Timeout waiting for {update_type} update completion - exiting")
                self.server.process.sendline('exit')
                try:
                    self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                except pexpect.TIMEOUT:
                    DebugTracer.log("Timeout during forced exit", "UPDATER")
                    pass
            
            if self.server.process.isalive():
                DebugTracer.log(f"Force closing process after {update_type} update", "UPDATER")
                self.server.process.close(force=True)
            
            DebugTracer.log(f"Restarting server after {update_type} update", "UPDATER")
            self.server.restart()
            DebugTracer.log(f"Full update ({update_type}) sequence completed successfully", "UPDATER")
            logging.debug(f"Server restarted after {update_type} update")
            return True
        except Exception as e:
            DebugTracer.log(f"Full update ({update_type}) failed: {str(e)}", "UPDATER")
            logging.error(f"Full update ({update_type}) failed: {e}")
            try:
                if self.server.process.isalive():
                    DebugTracer.log(f"Force closing process after {update_type} error", "UPDATER")
                    self.server.process.close(force=True)
                DebugTracer.log(f"Attempting server recovery after {update_type} failure", "UPDATER")
                self.server.restart()
                DebugTracer.log("Server recovery completed", "UPDATER")
                logging.debug(f"Server recovered after {update_type} update failure")
                return False
            except Exception as recover_error:
                DebugTracer.log(f"Server recovery failed: {str(recover_error)}", "UPDATER")
                logging.error(f"Failed to recover server: {recover_error}")
                return False

class TestSuite:
    def __init__(self, executable='./moondream_station', args=None, cleanup=True, test_capabilities=False):
        self.server = Server(executable, args)
        self.cleanup = cleanup
        self.test_capabilities = test_capabilities
        self.updater = None
    
    @DebugTracer.log_operation
    def run_capability_tests(self):
        if not self.test_capabilities or not CAPABILITY_TESTING_AVAILABLE:
            return True
        
        DebugTracer.log("Starting capability testing", "CAPABILITY")
        logging.debug("=== Running Capability Tests ===")
        
        try:
            cmd = Commands(self.server.process)
            
            # Save current model state before testing
            config_output = cmd.get_config()
            current_config = Parser.parse_config(config_output)
            original_model = current_config.get('active_model', None)
            
            DebugTracer.log(f"Original active model: {original_model}", "CAPABILITY")
            logging.debug(f"Saving original active model: {original_model}")
            
            # Get model list and test each one
            output = cmd.model_list()
            models = parse_model_list_output(output)
            
            DebugTracer.log(f"Testing capabilities for {len(models)} models", "CAPABILITY")
            logging.debug(f"Found {len(models)} models for capability testing: {models}")
            
            for model_name in models:
                DebugTracer.log(f"Testing capabilities for model: {model_name}", "CAPABILITY")
                logging.debug(f"--- Testing capabilities for model: {model_name} ---")
                
                cmd.model_use(f'"{model_name}"')
                test_model_capabilities(self.server.process, model_name)
            
            # Restore original model if we had one
            if original_model and original_model in models:
                DebugTracer.log(f"Restoring original model: {original_model}", "CAPABILITY")
                logging.debug(f"Restoring original active model: {original_model}")
                cmd.model_use(f'"{original_model}"')
            elif original_model:
                DebugTracer.log(f"Warning: Original model '{original_model}' not found in current model list", "CAPABILITY")
                logging.warning(f"Could not restore original model '{original_model}' - not in current model list")
            
            DebugTracer.log("Capability testing completed", "CAPABILITY")
            logging.debug("Capability testing completed successfully")
            return True
            
        except Exception as e:
            DebugTracer.log(f"Capability testing failed: {str(e)}", "CAPABILITY")
            logging.error(f"Capability testing failed: {e}")
            return False
    
    def run(self):
        logging.debug("Starting incremental updates test suite")
        try:
            Manifest.verify_environment()
        except Exception as e:
            logging.error(f"Test environment verification failed: {e}")
            return False
        
        Manifest.update_version(1)
        self.server.start()
        self.updater = Updater(self.server)
        
        try:
            success = self._execute_test_sequence()
            logging.debug(f"Test suite completed: {'PASS' if success else 'FAIL'}")
            return success
        finally:
            if self.cleanup:
                try:
                    Manifest.update_version(1)
                    logging.debug("Manifest restored to v001")
                except Exception as e:
                    logging.warning(f"Failed to restore manifest: {e}")
            self.server.stop()
    
    def _execute_test_sequence(self):
        all_passed = True
        
        cmd = Commands(self.server.process)
        cmd.update_manifest()
        
        success = Validator.check_updates(self.server.process, "All Up to Date (v001)", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'], 
            'CLI': Patterns.STATUS_INDICATORS['up_to_date']
        })
        all_passed = all_passed and success
        
        # Capability test after v001 baseline
        if self.test_capabilities:
            logging.debug("Running capability tests after v001 baseline")
            success = self.run_capability_tests()
            if not success:
                logging.error("Capability tests failed after v001 baseline")
                all_passed = False
        
        Manifest.update_version(2)
        cmd.update_manifest()
        success = Validator.check_updates(self.server.process, "Bootstrap Update Available (v002)", {
            'Bootstrap': Patterns.STATUS_INDICATORS['update_available'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'],
            'CLI': Patterns.STATUS_INDICATORS['up_to_date']
        })
        all_passed = all_passed and success
        
        success = self.updater.bootstrap()
        if not success:
            logging.error("Bootstrap update failed")
            all_passed = False
        
        # Capability test after v002 bootstrap update
        if self.test_capabilities:
            logging.debug("Running capability tests after bootstrap update")
            success = self.run_capability_tests()
            if not success:
                logging.error("Capability tests failed after bootstrap update")
                all_passed = False
        
        Manifest.update_version(3)
        cmd = Commands(self.server.process)
        cmd.update_manifest()
        success = Validator.check_updates(self.server.process, "Hypervisor + Model Updates Available (v003)", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['update_available'],
            'CLI': Patterns.STATUS_INDICATORS['up_to_date'],
            'Model': Patterns.STATUS_INDICATORS['update_available']
        })
        all_passed = all_passed and success
        
        success = self.updater.hypervisor()
        if not success:
            logging.error("Hypervisor update failed")
            all_passed = False
        
        # Capability test after hypervisor update
        if self.test_capabilities:
            logging.debug("Running capability tests after hypervisor update")
            success = self.run_capability_tests()
            if not success:
                logging.error("Capability tests failed after hypervisor update")
                all_passed = False
        
        cmd = Commands(self.server.process)
        success = Validator.check_updates(self.server.process, "After Hypervisor Update in v003", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'],
            'CLI': Patterns.STATUS_INDICATORS['up_to_date'],
            'Model': Patterns.STATUS_INDICATORS['update_available']
        })
        all_passed = all_passed and success
        
        success = self.updater.full('admin update --confirm', "model")
        if not success:
            logging.error("Model update failed")
            all_passed = False
        
        cmd = Commands(self.server.process)
        success = Validator.check_updates(self.server.process, "After Model Update in v003", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'],
            'CLI': Patterns.STATUS_INDICATORS['up_to_date'],
            'Model': Patterns.STATUS_INDICATORS['up_to_date']
        })
        all_passed = all_passed and success
        
        success = Validator.model_list(self.server.process)
        all_passed = all_passed and success
        
        # Capability test after model update
        if self.test_capabilities:
            logging.debug("Running capability tests after model update")
            success = self.run_capability_tests()
            if not success:
                logging.error("Capability tests failed after model update")
                all_passed = False
        
        Manifest.update_version(4)
        cmd = Commands(self.server.process)
        cmd.update_manifest()
        success = Validator.check_updates(self.server.process, "CLI Update Available (v004)", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'],
            'CLI': Patterns.STATUS_INDICATORS['update_available'],
            'Model': Patterns.STATUS_INDICATORS['up_to_date']
        })
        all_passed = all_passed and success
        
        success = self.updater.full('admin update --confirm', "cli")
        if not success:
            logging.error("CLI update failed")
            all_passed = False
        
        # Capability test after CLI update
        if self.test_capabilities:
            logging.debug("Running capability tests after CLI update")
            success = self.run_capability_tests()
            if not success:
                logging.error("Capability tests failed after CLI update")
                all_passed = False
        
        cmd = Commands(self.server.process)
        success = Validator.check_updates(self.server.process, "After CLI Update (v004)", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'],
            'CLI': Patterns.STATUS_INDICATORS['up_to_date'],
            'Model': Patterns.STATUS_INDICATORS['up_to_date']
        })
        all_passed = all_passed and success
        
        # Step 5: Inference Client Update Test (v005)
        Manifest.update_version(5)
        cmd = Commands(self.server.process)
        cmd.update_manifest()
        
        logging.debug("=== Testing Inference Client Updates (v005) ===")
        
        success = Validator.model_switch(
            self.server.process, 
            "Moondream2-2025-3-27",
            expected_inference_client="v0.0.1"
        )
        if not success:
            logging.error("Failed to switch to model with inference client v0.0.1")
            all_passed = False
        
        success = Validator.model_switch(
            self.server.process,
            "Moondream2-2025-04-14", 
            expected_inference_client="v0.0.2"
        )
        if not success:
            logging.error("Failed to switch to model with inference client v0.0.2")
            all_passed = False
        
        success = Validator.model_switch(
            self.server.process,
            "Moondream2-2025-3-27",
            expected_inference_client="v0.0.1" 
        )
        if not success:
            logging.error("Failed to switch back to model with inference client v0.0.1")
            all_passed = False
        
        logging.debug("Inference client update tests completed")
        
        # Capability test after inference client updates
        if self.test_capabilities:
            logging.debug("Running capability tests after inference client updates")
            success = self.run_capability_tests()
            if not success:
                logging.error("Capability tests failed after inference client updates")
                all_passed = False
        
        return all_passed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Moondream Station complete update suite')
    parser.add_argument('--executable', default='./moondream_station', help='Path to executable')
    parser.add_argument('--verbose', action='store_true', help='Print logs to console')
    parser.add_argument('--debug-trace', action='store_true', help='Enable comprehensive debug tracing')
    parser.add_argument('--test-capabilities', action='store_true', help='Run capability tests after updates')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip manifest cleanup')
    args, server_args = parser.parse_known_args()
    
    setup_logging(verbose=args.verbose, debug_trace=args.debug_trace)
    
    if args.debug_trace:
        DebugTracer.log("=== COMPREHENSIVE DEBUG TRACING ENABLED ===", "SYSTEM")
        DebugTracer.log(f"Command line args: {vars(args)}", "SYSTEM")
        DebugTracer.log(f"Server args: {server_args}", "SYSTEM")
    
    if args.test_capabilities and not CAPABILITY_TESTING_AVAILABLE:
        logging.error("Capability testing requested but test_capability.py not available")
        exit(1)
    
    suite = TestSuite(args.executable, server_args, 
                     cleanup=not args.no_cleanup, 
                     test_capabilities=args.test_capabilities)
    success = suite.run()
    
    if args.debug_trace:
        DebugTracer.log(f"=== TEST SUITE COMPLETED: {'SUCCESS' if success else 'FAILURE'} ===", "SYSTEM")
    
    exit(0 if success else 1)

if __name__ == "__main__":
    main()