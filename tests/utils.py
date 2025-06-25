import os
import re
import json
import logging
import shutil
import requests
import re

def clean_files(folder="$HOME/.local/share/MoondreamStation"):
    folder = os.path.expanduser(folder)
    if os.path.exists(folder):
        logging.debug(f"Attempting to clean folder...{folder}")
        shutil.rmtree(folder)
        logging.debug(f"Successfully cleaned {folder}")

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
    
    if command_type in ['caption', 'query']:

        filtered = [line for line in lines 
                   if not line.startswith(f'{command_type} ') 
                   and 'Generating' not in line
                   and 'Answering streaming query' not in line
                   and '------ Completed ------' not in line
                   and 'moondream>' not in line]
        return max(filtered, key=len) if filtered else output.strip()
    
    elif command_type == 'detect':
        if any('No' in line and 'detected' in line for line in lines):
            return "No face objects detected"
        match = re.search(r"Position: (\{[^}]+\})", output)
        return match.group(1) if match else output.strip()
    
    elif command_type == 'point':
        match = re.search(r"(\{'x': [^}]+\})", output)
        return match.group(1) if match else output.strip()
    
    return output.strip()

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