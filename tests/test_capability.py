import pexpect
import logging
import argparse
from utils import clean_files, load_expected_responses, clean_response_output, validate_model_list
# Timeout configurations
QUICK_TIMEOUT = 10
STANDARD_TIMEOUT = 30
LONG_TIMEOUT = 120
IMAGE_URL = "https://raw.githubusercontent.com/m87-labs/moondream-station/refs/heads/main/assets/md_logo_clean.png"

def setup_logging(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler('test_capabilities.log', mode='w')
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
    child = pexpect.spawn(' '.join(cmd))
    logging.debug(f"Starting up Moondream Station with command: {' '.join(cmd)}")
    child.expect('moondream>', timeout=STANDARD_TIMEOUT)
    return child

def end_server(child):
    # just check for exit message, don't wait for process to die, since bad shutdown in a known bug.
    child.sendline('exit')
    child.expect(r'Exiting Moondream CLI', timeout=QUICK_TIMEOUT)
    
    if child.isalive():
        child.close(force=True)

def check_health(child):
    child.sendline('health')
    child.expect('moondream>', timeout=STANDARD_TIMEOUT)
    health_prompt = child.before.decode()
    logging.debug("Health Check.")
    logging.debug(health_prompt)
    return child

def parse_model_list_output(output):
    models = []
    for line in output.split('\n'):
        line = line.strip()
        if line.startswith('Model: '):
            models.append(line[7:].strip())
    return models

def test_capability(child, command, expected_response, timeout=STANDARD_TIMEOUT):
    logging.debug(f"Testing: {command}")
    child.sendline(command)
    
    if 'caption' in command:
        try:
            child.expect('Generating streaming caption...', timeout=QUICK_TIMEOUT)
        except:
            pass
    
    child.expect('moondream>', timeout=timeout)
    
    output = child.before.decode().strip()
    cmd_type = next((t for t in ['caption', 'query', 'detect', 'point'] if t in command), None)
    cleaned = clean_response_output(output, cmd_type)
    
    # This chunk will validate length for captions
    length_valid = True
    if cmd_type == 'caption':
        words, chars = cleaned.split(), len(cleaned)
        length_type = next((t for t in ['short', 'long'] if f'--length {t}' in command), 'normal')
        cfg = load_expected_responses().get('caption_length_ranges', {}).get(length_type, {})
        
        length_valid = (cfg.get('min_words', 0) <= len(words) <= (cfg.get('max_words') or float('inf')) and
                       cfg.get('min_chars', 0) <= chars <= (cfg.get('max_chars') or float('inf')))
        
        logging.debug(f"Length {length_type}: {len(words)} words, {chars} chars "
                     f"({'PASS' if length_valid else 'FAIL'})")
    
    if cmd_type in ['caption', 'query']:
        keywords = expected_response.get('keywords', [])
        matches = sum(kw.lower() in cleaned.lower() for kw in keywords)
        content_valid = matches >= len(keywords) * 0.7
        logging.debug(f"Keywords: {matches}/{len(keywords)} ({'PASS' if content_valid else 'FAIL'})")
    else:
        content_valid = cleaned == expected_response
        logging.debug(f"Exact match: {'PASS' if content_valid else 'FAIL'}")
    
    logging.debug(f"Expected: {expected_response}\nGot: {cleaned}")
    
    return content_valid and length_valid, cleaned

def test_model_capabilities(child, model_name):
    expected_responses = load_expected_responses()
    if model_name not in expected_responses:
        logging.error(f"No expected responses found for model: {model_name}")
        return child
    
    model_expected = expected_responses[model_name]
    
    capabilities = [
        {
            'command': f'caption {IMAGE_URL} --length short',
            'expected': model_expected['caption_short'],
            'name': 'Caption Short'
        },
        {
            'command': f'caption {IMAGE_URL} --length normal',
            'expected': model_expected['caption_normal'],
            'name': 'Caption Normal'
        },
        {
            'command': f'caption {IMAGE_URL} --length long',
            'expected': model_expected['caption_long'],
            'name': 'Caption Long'
        },
        {
            'command': f'query "What is in this image?" {IMAGE_URL}',
            'expected': model_expected['query'],
            'name': 'Query'
        },
        {
            'command': f'detect face {IMAGE_URL}',
            'expected': model_expected['detect'],
            'name': 'Detect'
        },
        {
            'command': f'point face {IMAGE_URL}',
            'expected': model_expected['point'],
            'name': 'Point'
        }
    ]
    
    results = {}
    for i, cap in enumerate(capabilities):
        logging.debug("")
        logging.debug(f"--- {cap['name']} Test ---")
        try:
            success, output = test_capability(child, cap['command'], cap['expected'])
            results[cap['name']] = {'success': success, 'output': output}
        except Exception as e:
            logging.error(f"{cap['name']} test failed: {e}")
            logging.debug(f"")
            results[cap['name']] = {'success': False, 'output': str(e)}
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    logging.debug(f"Model capability tests: {passed}/{total} passed for {model_name}")
    
    return child

def test_all_models(child):
    child.sendline('admin model-list')
    child.expect('moondream>', timeout=STANDARD_TIMEOUT)
    model_list_output = child.before.decode()
    logging.debug("Model list output:")
    logging.debug(model_list_output)
    
    validate_model_list(model_list_output)
    
    models = parse_model_list_output(model_list_output)
    logging.debug(f"Found {len(models)} models: {models}")
    
    for model_name in models:
        logging.debug("")
        logging.debug(f"--- Testing model: {model_name} ---")
        
        child.sendline(f'admin model-use "{model_name}" --confirm')
        try:
            child.expect('Model initialization completed successfully!', timeout=LONG_TIMEOUT)
            child.expect('moondream>', timeout=QUICK_TIMEOUT)
            logging.debug(f"Successfully switched to model: {model_name}")
            logging.debug("")
            
            child = test_model_capabilities(child, model_name)
            
        except Exception as e:
            logging.warning(f"Model switch to '{model_name}' failed: {e}")
    
    return child

def test_server(cleanup=True, executable_path='./moondream_station', server_args=None):
    if cleanup:
        clean_files()
    child = start_server(executable_path, server_args)
    child = check_health(child)
    child = test_all_models(child)
    
    end_server(child)

def main():
    parser = argparse.ArgumentParser(description='Test Moondream Station startup')
    parser.add_argument('--no-cleanup', action='store_true', help='Skip cleanup before test')
    parser.add_argument('--executable', default='./moondream_station', help='Path to moondream_station executable')
    parser.add_argument('--verbose', action='store_true', help='Print log messages to console')
    
    args, server_args = parser.parse_known_args()
    
    setup_logging(verbose=args.verbose)
    
    test_server(cleanup=not args.no_cleanup, executable_path=args.executable, server_args=server_args)

if __name__ == "__main__":
    main()