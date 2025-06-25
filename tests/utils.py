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

def validate_model_list(model_list_output, manifest_url):
    try:
        if manifest_url.startswith(('http://', 'https://')):
            manifest_data = requests.get(manifest_url, timeout=10).json()
            logging.debug(f"Fetched manifest from: {manifest_url}")
        elif manifest_url.startswith('file://'):
            path = manifest_url[7:]  # Remove 'file://' prefix
            with open(os.path.expanduser(path), 'r') as f:
                manifest_data = json.load(f)
            logging.debug(f"Loaded manifest from file URL: {manifest_url}")
        else:
            with open(os.path.expanduser(manifest_url), 'r') as f:
                manifest_data = json.load(f)
            logging.debug(f"Loaded manifest from local file: {manifest_url}")
    except Exception as e:
        logging.warning(f"Manifest validation skipped: {e}")
        return
    
    cli_models = parse_model_list(model_list_output)
    manifest_models = manifest_data.get('models', {}).get('2b', {})
    
    logging.debug("--- Model List Validation ---")
    
    # Check all models match
    all_valid = True
    for name, cli_data in cli_models.items():
        if name not in manifest_models:
            logging.warning(f"Model '{name}' found in CLI but not in manifest")
            all_valid = False
            continue
            
        # Compare fields
        manifest_model = manifest_models[name]
        mismatches = [field for field in ['release_date', 'model_size', 'notes']
                      if cli_data.get(field) != manifest_model.get(field)]
        
        if mismatches:
            all_valid = False
            logging.debug(f"Model '{name}': FAIL")
            for field in mismatches:
                logging.debug(f"  {field}: expected '{manifest_model.get(field)}', got '{cli_data.get(field)}'")
        else:
            logging.debug(f"Model '{name}': PASS")
    
    # Check for missing models
    missing = set(manifest_models) - set(cli_models)
    for name in missing:
        logging.warning(f"Model '{name}' in manifest but not in CLI output")
        all_valid = False
    
    logging.debug(f"Model list validation: {'PASS' if all_valid else 'FAIL'}")