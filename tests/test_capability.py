import pexpect
import logging
import argparse
from contextlib import contextmanager
from utils import clean_files, load_expected_responses, clean_response_output, validate_model_list, parse_model_list

# Timeout configurations
QUICK_TIMEOUT = 60
STANDARD_TIMEOUT = 100
LONG_TIMEOUT = 120
KEYWORD_THRESHOLD = 0.7

# URL Default Constants
IMAGE_URL = "https://raw.githubusercontent.com/m87-labs/moondream-station/refs/heads/main/assets/md_logo_clean.png"

DEFAULT_MANIFEST_URL = "https://depot.moondream.ai/station/md_station_manifest_ubuntu.json"

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

@contextmanager
def server_session(executable_path='./moondream_station', args=None):
    """
    Context manager to start and stop the Moondream Station CLI.
    This spawns the CLI process, performs a health check,
    and yields the child process for further interaction.
    After the block, it sends an exit command and cleans up the process.
    If the process is still alive after sending 'exit', it is forcefully closed.
    """
    cmd = [executable_path] + (args or [])
    child = pexpect.spawn(' '.join(cmd))
    logging.debug(f"Starting up Moondream Station with command: {' '.join(cmd)}")
    child.expect('moondream>', timeout=STANDARD_TIMEOUT)

    child.sendline('health')
    child.expect('moondream>', timeout=STANDARD_TIMEOUT)
    logging.debug(f"Health Check.\n{child.before.decode()}")
    
    try:
        yield child
    finally:

        child.sendline('exit')
        child.expect(r'Exiting Moondream CLI', timeout=QUICK_TIMEOUT)
        child.isalive() and child.close(force=True)

def test_capability(child, command, expected_response, timeout=STANDARD_TIMEOUT):
    """
    Test a specific capability of the Moondream Station CLI.
    This works by sending a command to the CLI, waiting for the response,
    and then validating the output against expected values.
    It passes if the output matches the expected response,
    and if the length of the response is within expected ranges for captions.
    If the command is a caption command, it also checks the length of the caption.
    If the command is a caption or query, it checks for keyword presence.
    If the command is a point or detect command, it checks for exact match for coordinates.
    """
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
        content_valid = matches >= len(keywords) * KEYWORD_THRESHOLD
        logging.debug(f"Keywords: {matches}/{len(keywords)} (threshold: {KEYWORD_THRESHOLD}) "
                     f"({'PASS' if content_valid else 'FAIL'})")
    else:
        content_valid = cleaned == expected_response
        logging.debug(f"Exact match: {'PASS' if content_valid else 'FAIL'}")
    
    logging.debug(f"Expected: {expected_response}\nGot: {cleaned}")
    
    return content_valid and length_valid, cleaned

def test_model_capabilities(child, model_name):
    """
    Test the capabilities of a specific model by running predefined commands
    and comparing the outputs against expected responses.
    This function retrieves expected responses for the model from a JSON file,
    constructs commands for various capabilities (captioning, querying, face detection),
    and executes them in the Moondream Station CLI using test_capability.
    It logs the results of each test and summarizes the overall success rate.
    """
    expected_responses = load_expected_responses()
    if model_name not in expected_responses:
        logging.error(f"No expected responses found for model: {model_name}")
        return
    
    model_expected = expected_responses[model_name]
    
    capabilities = [
        (f'caption {IMAGE_URL} --length {l}', model_expected[f'caption_{l}'], f'Caption {l.title()}')
        for l in ['short', 'normal', 'long']
    ] + [
        (f'query "What is in this image?" {IMAGE_URL}', model_expected['query'], 'Query'),
        (f'detect face {IMAGE_URL}', model_expected['detect'], 'Detect'),
        (f'point face {IMAGE_URL}', model_expected['point'], 'Point')
    ]
    
    results = {}
    for cmd, expected, name in capabilities:
        logging.debug(f"\n--- {name} Test ---")
        try:
            success, output = test_capability(child, cmd, expected)
            results[name] = {'success': success, 'output': output}
        except Exception as e:
            logging.error(f"{name} test failed: {e}")
            results[name] = {'success': False, 'output': str(e)}
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    logging.debug(f"Model capability tests: {passed}/{total} passed for {model_name}")

def test_all_models(child, manifest_url=None):
    """
    Test all models available in the Moondream Station CLI.
    This function retrieves the list of models, iterates through each model,
    switches to the model using 'admin model-use', and tests its capabilities.
    It validates the model list against a manifest URL if provided,
    and logs the results of each model's capabilities tests.
    """
    child.sendline('admin model-list')
    child.expect('moondream>', timeout=STANDARD_TIMEOUT)
    model_list_output = child.before.decode()
    logging.debug(f"Model list output:\n{model_list_output}")
    
    if manifest_url:
        validate_model_list(model_list_output, manifest_url)
    
    models = list(parse_model_list(model_list_output).keys())
    logging.debug(f"Found {len(models)} models: {models}")
    
    for model_name in models:
        logging.debug(f"\n--- Testing model: {model_name} ---")
        
        child.sendline(f'admin model-use "{model_name}" --confirm')
        try:
            child.expect('Model initialization completed successfully!', timeout=LONG_TIMEOUT)
            child.expect('moondream>', timeout=QUICK_TIMEOUT)
            logging.debug(f"Successfully switched to model: {model_name}\n")
            
            test_model_capabilities(child, model_name)
            
        except Exception as e:
            logging.warning(f"Model switch to '{model_name}' failed: {e}")

def test_server(cleanup=False, executable_path='./moondream_station', server_args=None, 
                manifest_url=None, skip_validation=False):
    if cleanup:
        clean_files()
    
    if manifest_url:
        server_args = (server_args or []) + ['--manifest-url', manifest_url]
    
    with server_session(executable_path, server_args) as child:
        validation_url = None if skip_validation else manifest_url
        test_all_models(child, validation_url)

def main():
    parser = argparse.ArgumentParser(description='Test Moondream Station startup',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cleanup', action='store_true', help='Cleanup before test')
    parser.add_argument('--executable', default='./moondream_station', help='Path to moondream_station executable')
    parser.add_argument('--keyword-threshold', type=float, default=0.7,
                    help='Minimum fraction of keywords required (0.0-1.0, default: 0.7)')
    parser.add_argument('--verbose', action='store_true', help='Print log messages to console')
    parser.add_argument('--manifest-url', default=DEFAULT_MANIFEST_URL,
                        help='URL or local path to manifest.json - passed to executable and used for validation')
    parser.add_argument('--skip-validation', action='store_true', 
                        help='Skip manifest validation (URL still passed to executable)')
    
    args, server_args = parser.parse_known_args()
    global KEYWORD_THRESHOLD
    KEYWORD_THRESHOLD = args.keyword_threshold
    
    setup_logging(verbose=args.verbose)
    
    test_server(cleanup=args.cleanup,
            executable_path=args.executable, 
            server_args=server_args, 
            manifest_url=args.manifest_url,
            skip_validation=args.skip_validation)

if __name__ == "__main__":
    main()