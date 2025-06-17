import os
import json
import hashlib

def get_md5(filepath):
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except (IOError, OSError):
        return None

def scan_directory(root_path):
    result = {}
    for root, dirs, files in os.walk(root_path):
        for file in files:
            full_path = os.path.join(root, file)
            if os.path.isfile(full_path):
                rel_path = os.path.relpath(full_path, root_path).replace('\\', '/')
                md5_hash = get_md5(full_path)
                if md5_hash:
                    result[rel_path] = md5_hash
    return result

def validate_directory(dir_path, expected_json_path):
    if not os.path.exists(expected_json_path):
        return {'error': f'Expected JSON not found: {expected_json_path}', 'valid': False}
    
    if not os.path.exists(dir_path):
        return {'error': f'Directory not found: {dir_path}', 'valid': False}
    
    with open(expected_json_path, 'r') as f:
        expected = json.load(f)
    
    missing = []
    mismatched = []
    
    for rel_path, expected_hash in expected.items():
        full_path = os.path.join(dir_path, rel_path)
        if not os.path.isfile(full_path):
            missing.append(rel_path)
        elif get_md5(full_path) != expected_hash:
            mismatched.append(rel_path)
    
    return {
        'missing': missing, 
        'mismatched': mismatched, 
        'valid': len(missing) == 0 and len(mismatched) == 0,
        'total_expected': len(expected),
        'found': len(expected) - len(missing)
    }