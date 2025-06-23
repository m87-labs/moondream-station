import pexpect
import shutil
import os
import logging
import time
import requests
import re
from pathlib import Path

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

class Config:
    MANIFEST_PATH = os.path.expanduser("~/.local/share/MoondreamStation/manifest.py")
    MANIFEST_URL_PATTERN = r'MANIFEST_URL\s*=\s*["\']([^"\']+)["\']'
    MODEL_CATEGORY = '2b'

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

class Server:
    def __init__(self, executable='./moondream_station', args=None):
        self.executable = executable
        self.args = args or []
        self.process = None
    
    def start(self):
        cmd = [self.executable] + self.args
        logging.debug(f"Starting server: {' '.join(cmd)}")
        time.sleep(2)
        try:
            self.process = pexpect.spawn(' '.join(cmd))
            self.process.expect(Patterns.PROMPT, timeout=Timeouts.STARTUP)
            logging.debug("Server started successfully")
            return self.process
        except pexpect.EOF:
            output = self.process.before.decode() if self.process else 'None'
            logging.error(f"Server failed to start (EOF). Output: {output}")
            raise
        except pexpect.TIMEOUT:
            output = self.process.before.decode() if self.process else 'None'
            logging.error(f"Server startup timeout. Output: {output}")
            raise
    
    def stop(self):
        if not self.process:
            return
        try:
            self.process.sendline('exit')
            self.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
            if self.process.isalive():
                self.process.close(force=True)
            logging.debug("Server stopped successfully")
        except:
            if self.process.isalive():
                self.process.close(force=True)
            logging.debug("Server force stopped")
    
    def restart(self):
        logging.debug("Restarting server")
        self.stop()
        time.sleep(2)
        return self.start()

class Manifest:
    @staticmethod
    def update_version(version):
        manifest_file = Path(MANIFEST_DIR) / "manifest.json"
        version_file = Path(MANIFEST_DIR) / f"manifest_v{version:03d}.json"
        if not version_file.exists():
            raise FileNotFoundError(f"Version manifest {version_file} not found")
        shutil.copy2(version_file, manifest_file)
        logging.debug(f"Updated manifest.json to version {version:03d}")
    
    @staticmethod
    def verify_environment():
        manifest_dir = Path(MANIFEST_DIR)
        if not manifest_dir.exists():
            raise FileNotFoundError(f"Manifest directory {MANIFEST_DIR} not found")
        required = ['manifest_v001.json', 'manifest_v002.json', 'manifest_v003.json', 'manifest_v004.json']
        missing = [f for f in required if not (manifest_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing manifest files: {missing}")
        logging.debug("Test environment verified")

class Commands:
    def __init__(self, process):
        self.process = process
    
    def run(self, command, expect_pattern=None, timeout=Timeouts.STANDARD, expect_exit=False):
        logging.debug(f"Running: {command}")
        self.process.sendline(command)
        
        if expect_exit:
            try:
                if expect_pattern:
                    self.process.expect(expect_pattern, timeout=timeout)
                    logging.debug(f"Found expected pattern: {expect_pattern}")
                time.sleep(3)
                output = self.process.before.decode().strip() if hasattr(self.process, 'before') else ""
                logging.debug("Server exited as expected")
                return output
            except pexpect.EOF:
                output = self.process.before.decode().strip() if hasattr(self.process, 'before') else ""
                logging.debug("Server exited (EOF)")
                return output
            except pexpect.TIMEOUT:
                logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
                output = self.process.before.decode().strip() if hasattr(self.process, 'before') else ""
                return output
        else:
            if expect_pattern:
                try:
                    self.process.expect(expect_pattern, timeout=timeout)
                except pexpect.TIMEOUT:
                    logging.warning(f"Timeout waiting for pattern: {expect_pattern}")
            self.process.expect(Patterns.PROMPT, timeout=timeout)
            output = self.process.before.decode().strip()
            logging.debug(f"Command output: {output}")
            return output
    
    def update_manifest(self):
        return self.run('admin update-manifest', timeout=Timeouts.STANDARD)
    
    def check_updates(self):
        return self.run('admin check-updates')
    
    def model_list(self):
        return self.run('admin model-list')

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

class Validator:
    @staticmethod
    def check_updates(process, scenario, expected):
        logging.debug(f"=== {scenario} ===")
        cmd = Commands(process)
        output = cmd.check_updates()
        actual = Parser.parse_updates(output)
        
        success = True
        for component, expected_status in expected.items():
            actual_status = actual.get(component)
            if actual_status == expected_status:
                logging.debug(f"{component}: {actual_status}")
            else:
                logging.error(f"{component}: got '{actual_status}', expected '{expected_status}'")
                success = False
        
        logging.debug(f"Check updates result: {'PASS' if success else 'FAIL'}")
        return success
    
    @staticmethod
    def model_list(process):
        logging.debug("=== Testing Model List ===")
        cmd = Commands(process)
        output = cmd.model_list()
        models = Parser.parse_models(output)
        
        if not os.path.exists(Config.MANIFEST_PATH):
            logging.warning("manifest.py not found - skipping model list validation")
            return True
        
        with open(Config.MANIFEST_PATH, 'r') as f:
            manifest_content = f.read()
        
        url_match = re.search(Config.MANIFEST_URL_PATTERN, manifest_content)
        if not url_match:
            logging.warning("MANIFEST_URL not found - skipping validation")
            return True
        
        try:
            response = requests.get(url_match.group(1), timeout=10)
            manifest_data = response.json()
            logging.debug(f"Fetched manifest from: {url_match.group(1)}")
        except Exception as e:
            logging.warning(f"Failed to fetch manifest: {e}")
            return True
        
        manifest_models = manifest_data.get('models', {}).get(Config.MODEL_CATEGORY, {})
        
        all_valid = True
        for model_name, cli_data in models.items():
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
            if model_name not in models:
                logging.warning(f"Model '{model_name}' in manifest but not in CLI output")
                all_valid = False
        
        logging.debug(f"Model list validation: {'PASS' if all_valid else 'FAIL'}")
        return all_valid

class Updater:
    def __init__(self, server):
        self.server = server
    
    def bootstrap(self, command='admin update-bootstrap --confirm'):
        logging.debug(f"Executing bootstrap update: {command}")
        try:
            cmd = Commands(self.server.process)
            cmd.run(command, expect_pattern=Patterns.UPDATE_COMPLETION['bootstrap'], 
                   timeout=Timeouts.UPDATE, expect_exit=True)
            logging.debug("Bootstrap update completed successfully")
            
            if self.server.process.isalive():
                self.server.process.close(force=True)
            
            self.server.restart()
            logging.debug("Server restarted after bootstrap update")
            return True
        except Exception as e:
            logging.error(f"Bootstrap update failed: {e}")
            try:
                if self.server.process.isalive():
                    self.server.process.close(force=True)
                time.sleep(3)
                self.server.restart()
                logging.debug("Server recovered after bootstrap update failure")
                return False
            except Exception as recover_error:
                logging.error(f"Failed to recover server: {recover_error}")
                return False
    
    def hypervisor(self, command='admin update-hypervisor --confirm'):
        logging.debug(f"Executing hypervisor update: {command}")
        try:
            self.server.process.sendline(command)
            try:
                index = self.server.process.expect([
                    Patterns.UPDATE_COMPLETION['hypervisor_complete'],
                    Patterns.UPDATE_COMPLETION['hypervisor_off'],
                    Patterns.PROMPT
                ], timeout=Timeouts.UPDATE)
                
                if index == 0:
                    logging.debug("Hypervisor update completed")
                    self.server.process.expect(Patterns.PROMPT, timeout=Timeouts.QUICK)
                elif index == 1:
                    logging.debug("Found 'Hypervisor: off' state - exiting as expected")
                    self.server.process.sendline('exit')
                    try:
                        self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                        logging.debug("Exited CLI after hypervisor update")
                    except (pexpect.TIMEOUT, pexpect.EOF):
                        logging.debug("CLI process ended during hypervisor update")
                else:
                    logging.warning("Hypervisor update returned to prompt unexpectedly")
            except pexpect.TIMEOUT:
                logging.warning("Timeout waiting for hypervisor update - exiting")
                self.server.process.sendline('exit')
                try:
                    self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                except pexpect.TIMEOUT:
                    pass
            
            if self.server.process.isalive():
                self.server.process.close(force=True)
            
            self.server.restart()
            logging.debug("Server restarted after hypervisor update")
            return True
        except Exception as e:
            logging.error(f"Hypervisor update failed: {e}")
            try:
                if self.server.process.isalive():
                    self.server.process.close(force=True)
                self.server.restart()
                logging.debug("Server recovered after hypervisor update failure")
                return False
            except Exception as recover_error:
                logging.error(f"Failed to recover server: {recover_error}")
                return False
    
    def full(self, command='admin update --confirm', update_type="general"):
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
                index = self.server.process.expect([completion_pattern, Patterns.PROMPT], timeout=Timeouts.UPDATE)
                
                if index == 0:
                    logging.debug(f"Found completion message for {update_type} update - exiting")
                    self.server.process.sendline('exit')
                    try:
                        self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                        logging.debug(f"Exited CLI after {update_type} update")
                    except (pexpect.TIMEOUT, pexpect.EOF):
                        logging.debug(f"CLI process ended during {update_type} update")
                else:
                    logging.warning(f"Full update ({update_type}) returned to prompt unexpectedly")
                    
            except pexpect.TIMEOUT:
                logging.warning(f"Timeout waiting for {update_type} update completion - exiting")
                self.server.process.sendline('exit')
                try:
                    self.server.process.expect(Patterns.EXIT_MESSAGE, timeout=Timeouts.QUICK)
                except pexpect.TIMEOUT:
                    pass
            
            if self.server.process.isalive():
                self.server.process.close(force=True)
            
            self.server.restart()
            logging.debug(f"Server restarted after {update_type} update")
            return True
        except Exception as e:
            logging.error(f"Full update ({update_type}) failed: {e}")
            try:
                if self.server.process.isalive():
                    self.server.process.close(force=True)
                self.server.restart()
                logging.debug(f"Server recovered after {update_type} update failure")
                return False
            except Exception as recover_error:
                logging.error(f"Failed to recover server: {recover_error}")
                return False

class TestSuite:
    def __init__(self, executable='./moondream_station', args=None, cleanup=True):
        self.server = Server(executable, args)
        self.cleanup = cleanup
        self.updater = None
    
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
        
        cmd = Commands(self.server.process)
        success = Validator.check_updates(self.server.process, "After CLI Update (v004)", {
            'Bootstrap': Patterns.STATUS_INDICATORS['up_to_date'],
            'Hypervisor': Patterns.STATUS_INDICATORS['up_to_date'],
            'CLI': Patterns.STATUS_INDICATORS['up_to_date'],
            'Model': Patterns.STATUS_INDICATORS['up_to_date']
        })
        all_passed = all_passed and success
        
        return all_passed

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Moondream Station complete update suite')
    parser.add_argument('--executable', default='./moondream_station', help='Path to executable')
    parser.add_argument('--verbose', action='store_true', help='Print logs to console')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip manifest cleanup')
    args, server_args = parser.parse_known_args()
    
    setup_logging(verbose=args.verbose)
    suite = TestSuite(args.executable, server_args, cleanup=not args.no_cleanup)
    success = suite.run()
    exit(0 if success else 1)

if __name__ == "__main__":
    main()