import os
import re
import json
import socket
import logging
import shutil
import psutil
import requests
import re
from verify_checksum_json import validate_directory

def clean_files(folder = "$HOME/.local/share/MoondreamStation"):
    folder = os.path.expanduser(folder)
    if os.path.exists(folder):
        logging.debug(f"Attempting to clean folder...{folder}")
        shutil.rmtree(folder)
    if not os.path.exists(folder):
        logging.debug(f"Successfully cleaned {folder}")
        return
    else:
        logging.debug(f"Folder was not cleaned.")

def load_expected_responses(json_path="expected_responses.json"):
    """Load expected responses from JSON file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load expected responses: {e}")
        return {}

def clean_response_output(output, command_type):
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    
    if command_type == 'caption':
        caption_lines = []
        for line in lines:
            if (not line.startswith('caption ') and 
                'Generating' not in line and
                '------ Completed ------' not in line and
                'moondream>' not in line):
                caption_lines.append(line)
        
        return max(caption_lines, key=len) if caption_lines else output.strip()
    
    elif command_type == 'query':
        query_lines = []
        for line in lines:
            if (not line.startswith('query ') and
                'Answering streaming query' not in line and
                '------ Completed ------' not in line and
                'moondream>' not in line):
                query_lines.append(line)
        
        return max(query_lines, key=len) if query_lines else output.strip()
    
    elif command_type == 'detect':
        if any('No' in line and 'detected' in line for line in lines):
            return "No face objects detected"
        
        for line in lines:
            match = re.search(r"Position: (\{[^}]+\})", line)
            if match:
                return match.group(1)
        return output.strip()
    
    elif command_type == 'point':
        for line in lines:
            match = re.search(r"(\{'x': [^}]+\})", line)
            if match:
                return match.group(1)
        return output.strip()
    
    else:
        return output.strip()

def validate_files(dir_path, expected_json):
    result = validate_directory(dir_path, expected_json)

    if result.get('error'):
        logging.debug(f"Validation error: {result['error']}")
        return result

    if result['valid']:
        logging.debug(f"File validation PASSED: {result['found']}/{result['total_expected']} files valid")
    else:
        missing_count = len(result.get('missing', []))
        mismatched_count = len(result.get('mismatched', []))

        if missing_count > 0 and mismatched_count > 0:
            logging.debug(f"File validation FAILED: {result['found']}/{result['total_expected']} files found, {missing_count} missing, {mismatched_count} mismatched")
        elif missing_count > 0:
            logging.debug(f"File validation FAILED: {result['found']}/{result['total_expected']} files found, {missing_count} missing")
        elif mismatched_count > 0:
            logging.debug(f"File validation FAILED: {result['found']}/{result['total_expected']} files found, {mismatched_count} hash mismatched")

        if result.get('missing'):
            logging.debug(f"Missing files: {result['missing']}")
        if result.get('mismatched'):
            logging.debug(f"Mismatched files: {result['mismatched']}")

    return result

def is_port_occupied(port, host='localhost'):
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       return s.connect_ex((host, port)) == 0
   
def get_port_process_pid(port):
    for proc in psutil.process_iter(['pid', 'connections']):
        try:
            connections = proc.info['connections'] or []
            for conn in connections:
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def get_process_start_time(pid):
    try:
        proc = psutil.Process(pid)
        return proc.create_time()
    except:
        return None

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
        return
    
    with open(manifest_path, 'r') as f:
        manifest_content = f.read()
    
    url_match = re.search(r'MANIFEST_URL\s*=\s*["\']([^"\']+)["\']', manifest_content)
    if not url_match:
        logging.warning("MANIFEST_URL not found in manifest.py - skipping validation")
        return
    
    try:
        response = requests.get(url_match.group(1), timeout=10)
        manifest_data = response.json()
        logging.debug(f"Fetched manifest from: {url_match.group(1)}")
    except Exception as e:
        logging.warning(f"Failed to fetch manifest: {e}")
        return
    
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