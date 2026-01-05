"""
MULTI-PLATFORM HOMEWORK SOLVER BOT
===================================
Automatically solves homework questions from Yaklass and Google Forms using AI.

Features:
  - Auto-detects platform (Yaklass, Google Forms)
  - Parallel AI querying (Perplexity + Groq, all available models)
  - Consensus voting with similarity matching
  - Dynamic model discovery (tests each model for validity)
  - Answer extraction and cleaning (removes citations, symbols)
  - Automatic submission and navigation
  - Fully recursive batch solving (loop through entire test)
  - Smart answer field detection (text input, select, radio, checkbox)
  - JSON logging with timestamps
  - Colorized console output with progress tracking

Setup:
  1. Create .env file with PERPLEXITY_API_KEY and GROQ_API_KEY
  2. Run script and press F8 on any question page (Yaklass or Google Forms)
  3. Bot will automatically detect platform and solve questions
  4. Works with both text answers and multiple choice (select/radio/checkbox)

Author: Bot
Version: 2.1 (Multi-platform: Yaklass + Google Forms)
"""
import threading
import time
import sys
import random
import keyboard
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from openai import OpenAI
from colorama import Fore, Style, init
from groq import Groq
import os
import json
import math
from datetime import datetime
from pathlib import Path
import concurrent.futures
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException, NoSuchElementException
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
init(autoreset=True)

# ==================== ENV LOADER & VALIDATOR ====================

def load_and_validate_env():
    """Load .env file and validate API keys. Simpler for users."""
    load_dotenv()  # Load .env if it exists
    
    print(f"\n{Fore.CYAN}[*] Checking API Configuration...")
    
    perplexity_key = os.getenv("PERPLEXITY_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    
    env_file = Path(".env")
    needs_update = False
    
    # Check Perplexity key
    if not perplexity_key:
        print(f"{Fore.YELLOW}[!] PERPLEXITY_API_KEY not found in .env")
        print(f"{Fore.CYAN}    Please add: PERPLEXITY_API_KEY=pplx-your-key-here")
        needs_update = True
    else:
        if len(perplexity_key) < 10:
            print(f"{Fore.RED}    ✗ PERPLEXITY_API_KEY too short (got {len(perplexity_key)} chars)")
            needs_update = True
        else:
            print(f"{Fore.GREEN}    ✓ PERPLEXITY_API_KEY found ({len(perplexity_key)} chars)")
    
    # Check Groq key
    if not groq_key:
        print(f"{Fore.YELLOW}[!] GROQ_API_KEY not found in .env")
        print(f"{Fore.CYAN}    Please add: GROQ_API_KEY=gsk-your-key-here")
        needs_update = True
    else:
        if len(groq_key) < 10:
            print(f"{Fore.RED}    ✗ GROQ_API_KEY too short (got {len(groq_key)} chars)")
            needs_update = True
        else:
            print(f"{Fore.GREEN}    ✓ GROQ_API_KEY found ({len(groq_key)} chars)")
    
    if needs_update:
        print(f"\n{Fore.YELLOW}[!] Please update your .env file with valid API keys")
        print(f"{Fore.CYAN}    1. Open .env file")
        print(f"{Fore.CYAN}    2. Add your API keys")
        print(f"{Fore.CYAN}    3. Save and run the bot again")
        return False, perplexity_key, groq_key
    
    print(f"{Fore.GREEN}[+] API keys validated!\n")
    return True, perplexity_key, groq_key

# Load and validate first
validated, pplx_key, groq_key = load_and_validate_env()
if not validated:
    input(f"{Fore.RED}Press Enter to exit...")
    exit()

# ==================== CONFIGURATION CONSTANTS ====================

HOTKEY = "F8"
CHROME_DEBUG_PORT = "127.0.0.1:9222"
TYPING_MIN_DELAY = 0.01
TYPING_MAX_DELAY = 0.03
PAGE_LOAD_DELAY = 0.5
SUBMIT_WAIT_TIME = 2
NAVIGATION_WAIT_TIME = 1
NEXT_PAGE_WAIT_TIME = 2
MAX_ANSWER_EXTRACT_WORDS = 5
MAX_QUESTION_PREVIEW_LENGTH = 60
MAX_LOGGED_CONTENT_LENGTH = 500
MAX_ERROR_MSG_LENGTH = 80

# Consensus configuration
REQUIRED_RATIO = float(os.getenv("REQUIRED_RATIO", "0.6"))
MIN_REQUIRED = int(os.getenv("MIN_REQUIRED", "3"))
PREFER_PERPLEXITY = os.getenv("PREFER_PERPLEXITY", "1") in ("1", "true", "True")
TOTAL_TIMEOUT = int(os.getenv("TOTAL_TIMEOUT", "25"))
QUERY_TIMEOUT = 20
MAX_QUERY_RETRIES = 2
INITIAL_BACKOFF = 0.3
MAX_BACKOFF = 0.9
BACKOFF_MULTIPLIER = 1.5

# Platform detection (set dynamically)
CURRENT_PLATFORM = None  # Will be 'yaklass' or 'google_forms'

perplexity_client = OpenAI(api_key=pplx_key, base_url="https://api.perplexity.ai")
groq_client = Groq(api_key=groq_key)

# Fetch available models from Groq
def fetch_groq_models():
    """Fetch all available text-to-text models from Groq API."""
    try:
        models = groq_client.models.list()
        model_names = [m.id for m in models.data if 'text' in m.id.lower() or m.id in [
            'mixtral-8x7b-32768', 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile', 'llama-2-70b-4096', 'gemma-7b-it'
        ]]
        return sorted(list(set(model_names)))
    except Exception as e:
        print(f"{Fore.YELLOW}[!] Could not fetch Groq models: {str(e)[:60]}")
        # Fallback to manual list
        return [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "reasoning-gpt-oss-120b",
        ]

# Available Perplexity models (API doesn't expose list endpoint, so we use known models)
PERPLEXITY_MODELS_KNOWN = [
    "sonar-pro",
    "sonar",
    "sonar-reasoning-pro",
    "sonar-reasoning",
]

GROQ_MODELS = fetch_groq_models()
PERPLEXITY_MODELS = PERPLEXITY_MODELS_KNOWN

MODELS = []
for model in PERPLEXITY_MODELS:
    MODELS.append(("perplexity", model))
for model in GROQ_MODELS:
    MODELS.append(("groq", model))

WORKING_MODELS = list(MODELS)  # Will be updated after API check

error_log = []
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Statistics tracking
stats = {
    "questions_solved": 0,
    "questions_failed": 0,
    "total_time": 0,
    "start_time": None,
}

# ==================== MODEL DISCOVERY & VALIDATION ====================

def discover_and_validate_models():
    """Discover all available models and test which ones work. Keep only working models."""
    global WORKING_MODELS
    
    print(f"\n{Fore.CYAN}[*] Discovering and validating all available models...")
    
    # Get all candidate models
    perplexity_candidates = [
        "sonar-pro",
        "sonar",
        "sonar-reasoning-pro",
        "sonar-reasoning",
    ]
    
    groq_candidates = []
    try:
        models = groq_client.models.list()
        groq_candidates = [m.id for m in models.data]
        print(f"{Fore.CYAN}    Fetched {len(groq_candidates)} Groq models from API")
    except Exception as e:
        print(f"{Fore.YELLOW}    [!] Could not fetch Groq model list: {str(e)[:60]}")
        groq_candidates = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile", 
            "reasoning-gpt-oss-120b",
            "llama-2-70b-4096",
            "gemma-7b-it",
            "mixtral-8x7b-32768",
        ]
    
    working_models = []
    
    # Test Perplexity models
    print(f"\n{Fore.CYAN}[*] Testing Perplexity models ({len(perplexity_candidates)} total)...")
    for model in perplexity_candidates:
        try:
            response = perplexity_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                temperature=0.3,
                max_tokens=10
            )
            print(f"{Fore.GREEN}    ✓ {model}")
            working_models.append(("perplexity", model))
        except Exception as e:
            print(f"{Fore.RED}    ✗ {model}: {str(e)[:50]}")
    
    # Test Groq models
    print(f"\n{Fore.CYAN}[*] Testing Groq models ({len(groq_candidates)} total)...")
    groq_working_count = 0
    for model in groq_candidates:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                temperature=0.3,
                max_tokens=10
            )
            print(f"{Fore.GREEN}    ✓ {model}")
            working_models.append(("groq", model))
            groq_working_count += 1
        except Exception:
            pass  # Silently skip non-working models
    
    if groq_working_count == 0:
        print(f"{Fore.YELLOW}    [!] No working Groq models found")
    
    if len(working_models) < 2:
        print(f"\n{Fore.RED}[!] Less than 2 working models found!")
        print(f"{Fore.RED}    Please check your API keys and internet connection.")
        return []
    
    print(f"\n{Fore.GREEN}[+] Found {len(working_models)} working models:")
    for provider, model in working_models:
        print(f"    {Fore.GREEN}✓ {provider.upper()}: {model}")
    
    WORKING_MODELS = working_models
    return working_models

print(f"{Fore.CYAN}[*] Connecting to Chrome Debugger...")
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

try:
    driver = webdriver.Chrome(options=chrome_options)
    print(f"{Fore.GREEN}[+] Connected successfully! Bot is ready.")
except Exception as e:
    print(f"{Fore.RED}[!] Could not connect to Chrome. Make sure 'Chrome Debug' is open.")
    print(f"{Fore.YELLOW}Details: {e}")
    input("Press Enter to exit...")
    exit()

def normalize_answer(text):
    return " ".join(text.lower().strip().split())

# ==================== PLATFORM DETECTION ====================

def detect_platform():
    """Detect which platform we're on (Yaklass or Google Forms)."""
    global CURRENT_PLATFORM
    try:
        current_url = driver.current_url.lower()
        
        if "yaklass" in current_url or "якласс" in current_url:
            CURRENT_PLATFORM = "yaklass"
            return "yaklass"
        elif "forms.google.com" in current_url or "google.com/forms" in current_url:
            CURRENT_PLATFORM = "google_forms"
            return "google_forms"
        else:
            # Try to detect by page structure
            try:
                # Yaklass has specific divs
                driver.find_element(By.CSS_SELECTOR, "div#taskhtml")
                CURRENT_PLATFORM = "yaklass"
                return "yaklass"
            except:
                try:
                    # Google Forms has form elements
                    driver.find_element(By.XPATH, "//div[@data-item-id]")
                    CURRENT_PLATFORM = "google_forms"
                    return "google_forms"
                except:
                    # Default to yaklass if unsure
                    CURRENT_PLATFORM = "yaklass"
                    return "yaklass"
    except Exception as e:
        print(f"{Fore.YELLOW}[!] Could not detect platform: {str(e)[:60]}")
        CURRENT_PLATFORM = "yaklass"
        return "yaklass"

def normalize_answer(text):
    return " ".join(text.lower().strip().split())

def clean_answer(text):
    """Remove markdown, citations, and special symbols. Keep only words."""
    import re
    # Remove markdown bold/italic
    text = re.sub(r'[*_]{1,3}', '', text)
    # Remove citations [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove <think> tags and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()

def extract_core_answer(text):
    """Extract the core 2-5 word answer from a full response."""
    import re
    # Clean first
    text = clean_answer(text)
    if not text:
        return ""
    
    # Split into sentences (period, newline, etc.)
    sentences = re.split(r'[.\n!?]+', text)
    
    # Get first sentence that has actual content
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 1:
            # Take first 2-5 words max from first sentence
            words = sentence.split()[:MAX_ANSWER_EXTRACT_WORDS]
            return " ".join(words)
    
    # Fallback: just return first few words
    words = text.split()[:MAX_ANSWER_EXTRACT_WORDS]
    return " ".join(words)

def similarity_score(ans1, ans2):
    norm1 = normalize_answer(ans1)
    norm2 = normalize_answer(ans2)
    
    if norm1 == norm2:
        return 1.0
    if norm1 in norm2 or norm2 in norm1:
        return 0.8
    
    words1 = set(norm1.split())
    words2 = set(norm2.split())
    if words1 and words2:
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap
    return 0.0

def query_perplexity(model, question):
    """
    Query Perplexity API with exponential backoff retry logic.
    
    Args:
        model (str): Model name (e.g., 'sonar-pro')
        question (str): Question text to answer
    
    Returns:
        str: Model's answer or None on failure
        
    Note:
        max_tokens=100: Allows complete responses for long Russian questions.
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_QUERY_RETRIES + 1):
        try:
            response = perplexity_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Output ONLY the direct answer in 2-3 words. No explanation."},
                    {"role": "user", "content": f"Answer: {question}"}
                ],
                temperature=0.2,
                max_tokens=100,
                timeout=QUERY_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            error_log.append(f"Perplexity {model} attempt {attempt}: {msg[:MAX_ERROR_MSG_LENGTH]}")
            if attempt < MAX_QUERY_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
    return None

def query_groq(model, question):
    """
    Query Groq API with exponential backoff retry logic.
    
    Args:
        model (str): Model name (e.g., 'llama-3.3-70b-versatile')
        question (str): Question text to answer
    
    Returns:
        str: Model's answer or None on failure
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_QUERY_RETRIES + 1):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Output ONLY the direct answer in 2-3 words. No explanation."},
                    {"role": "user", "content": f"Answer: {question}"}
                ],
                temperature=0.2,
                max_tokens=100,
                timeout=QUERY_TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            error_log.append(f"Groq {model} attempt {attempt}: {msg[:MAX_ERROR_MSG_LENGTH]}")
            if attempt < MAX_QUERY_RETRIES:
                time.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
    return None

def get_answers_from_models(question, max_workers=5):
    answers = []
    responses = {}
    print(f"{Fore.CYAN}[*] Querying {len(WORKING_MODELS)} AI models (parallel)...")

    def worker(provider, model):
        model_display = f"{provider.upper()}:{model}"
        try:
            print(f"{Fore.YELLOW}    ↳ {model_display}...", flush=True)
            if provider == "perplexity":
                res = query_perplexity(model, question)
            elif provider == "groq":
                res = query_groq(model, question)
            else:
                res = None
            if res:
                # Extract core 2-3 word answer
                core = extract_core_answer(res)
                print(f"    {Fore.GREEN}✓ {model_display}: {core}")
                return (model_display, core, None)
            else:
                print(f"    {Fore.RED}✗ {model_display}: No answer")
                return (model_display, None, f"No answer from {model_display}")
        except Exception as e:
            msg = str(e)
            error_log.append(f"Query {provider}/{model} error: {msg[:240]}")
            return (model_display, None, msg)

    # Adaptive timeout: 15-20s based on model count
    timeout = min(20, max(15, len(WORKING_MODELS) * 1.5))
    print(f"{Fore.CYAN}[*] Waiting for responses (timeout: {int(timeout)}s)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, provider, model) for provider, model in WORKING_MODELS]
        done, not_done = concurrent.futures.wait(futures, timeout=timeout)

        if not_done:
            error_log.append(f"Timeout: {len(not_done)} model(s) slow")

        for fut in done:
            try:
                model_display, ans, err = fut.result()
            except Exception as e:
                error_log.append(f"Worker exception: {str(e)[:240]}")
                continue
            responses[model_display] = {"answer": ans, "error": err}

    # Collect answers in order
    for provider, model in WORKING_MODELS:
        key = f"{provider.upper()}:{model}"
        if key in responses and responses[key]["answer"]:
            answers.append((key, responses[key]["answer"]))
    
    print(f"{Fore.GREEN}[+] Got {len(answers)}/{len(WORKING_MODELS)} answers.")
    return answers

def verify_and_select_answer(answers, required_matches=None):
    if not answers:
        print(f"{Fore.RED}[!] No answers received!")
        return None
    
    print(f"\n{Fore.CYAN}[*] Analyzing {len(answers)} answer(s)...")
    best_answer = None
    best_match_count = 0
    counts = {}
    per_provider_counts = {}

    for model1, ans1 in answers:
        # Already extracted, just use directly
        match_count = sum(1 for _, ans2 in answers if similarity_score(ans1, ans2) >= 0.7)
        print(f"    {model1}: '{ans1}' (matches: {match_count}/{len(answers)})")
        norm = normalize_answer(ans1)
        counts.setdefault(norm, 0)
        counts[norm] = max(counts[norm], match_count)

        # provider-specific votes
        provider = model1.split(":")[0].lower()
        per_provider_counts.setdefault(norm, {})
        per_provider_counts[norm].setdefault(provider, 0)
        per_provider_counts[norm][provider] += 1

        if match_count > best_match_count:
            best_match_count = match_count
            best_answer = ans1

    total_models = len(answers)
    if required_matches is None:
        required_matches = max(MIN_REQUIRED, math.ceil(REQUIRED_RATIO * total_models))
        required_matches = min(required_matches, total_models)

    # Find consensus answer
    for norm_ans, cnt in counts.items():
        if cnt >= required_matches:
            candidate = next(a for _, a in answers if normalize_answer(a) == norm_ans)
            print(f"{Fore.GREEN}[+] ✓ CONSENSUS (>= {required_matches}/{total_models}): {candidate}")
            return candidate

    # No consensus - fallback to Perplexity if enabled
    print(f"{Fore.YELLOW}[!] No consensus ({required_matches}). Fallback...")
    if PREFER_PERPLEXITY:
        best_norm = None
        best_perf = -1
        for norm_ans, pdata in per_provider_counts.items():
            perf = pdata.get('perplexity', 0)
            if perf > best_perf:
                best_perf = perf
                best_norm = norm_ans
        if best_norm:
            candidate = next(a for _, a in answers if normalize_answer(a) == best_norm)
            print(f"{Fore.YELLOW}[!] Using Perplexity: {candidate}")
            return candidate

    # Final fallback: highest match count
    if best_answer:
        print(f"{Fore.YELLOW}[!] Using best-match: {best_answer}")
        return best_answer
    return answers[0][1]

def human_type(element, text):
    """Type text character by character with human-like delays."""
    try:
        element.clear()
    except Exception:
        pass
    
    for char in text:
        element.send_keys(char)
        time.sleep(random.uniform(TYPING_MIN_DELAY, TYPING_MAX_DELAY))

def type_answer(answer, max_attempts=2):
    """
    Find answer field and type answer or select from options.
    Handles both text input and multiple choice (radio, checkbox, select).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            answer_field, field_type = find_answer_field()
            
            if answer_field is None:
                return False
            
            # Handle text input
            if field_type == 'text':
                try:
                    answer_field.click()
                except Exception:
                    pass
                
                human_type(answer_field, answer)
                
                # Quick verification
                try:
                    val = answer_field.get_attribute('value')
                    if val and normalize_answer(val).startswith(normalize_answer(answer)[:5]):
                        return True
                except Exception:
                    pass
                
                try:
                    text = answer_field.text
                    if text and normalize_answer(text).startswith(normalize_answer(answer)[:5]):
                        return True
                except Exception:
                    pass
            
            # Handle select dropdown
            elif field_type == 'select':
                return select_best_match_option(answer_field, answer)
            
            # Handle radio buttons
            elif field_type == 'radio':
                return select_best_match_option(answer_field, answer, is_radio=True)
            
            # Handle checkboxes
            elif field_type == 'checkbox':
                return select_best_match_option(answer_field, answer, is_checkbox=True)
            
            time.sleep(0.2)
        
        except StaleElementReferenceException:
            error_log.append(f"Stale element - retry {attempt}")
            time.sleep(0.15 * attempt)
            continue
        except Exception as e:
            error_log.append(f"Type error: {str(e)[:100]}")
            time.sleep(0.15 * attempt)
            continue
    
    return False

def select_best_match_option(options_element, answer, is_radio=False, is_checkbox=False):
    """
    Select the best matching option from multiple choice options.
    
    Args:
        options_element: List of option elements or select dropdown
        answer: The answer text to match
        is_radio: True if radio buttons
        is_checkbox: True if checkboxes
    
    Returns:
        bool: True if selection successful
    """
    try:
        answer_norm = normalize_answer(answer)
        
        if isinstance(options_element, list):
            # Radio or checkbox list
            best_match = None
            best_score = 0.0
            
            for option in options_element:
                try:
                    # Get option text
                    option_text = option.text.strip()
                    if not option_text:
                        option_text = option.get_attribute('value')
                    
                    if option_text:
                        score = similarity_score(answer, option_text)
                        if score > best_score:
                            best_score = score
                            best_match = option
                except:
                    pass
            
            if best_match and best_score > 0.3:
                best_match.click()
                time.sleep(0.3)
                return True
        
        else:
            # Select dropdown
            try:
                from selenium.webdriver.support.select import Select
                select = Select(options_element)
                
                best_match = None
                best_score = 0.0
                
                for option in select.options:
                    option_text = option.text.strip()
                    score = similarity_score(answer, option_text)
                    if score > best_score:
                        best_score = score
                        best_match = option
                
                if best_match and best_score > 0.3:
                    select.select_by_value(best_match.get_attribute('value'))
                    time.sleep(0.3)
                    return True
            except:
                pass
        
        return False
    
    except Exception as e:
        error_log.append(f"Option selection: {str(e)[:100]}")
        return False

def find_answer_field():
    """
    Find answer input field - supports both Yaklass and Google Forms.
    Returns: (field_element, field_type) where field_type is 'text', 'select', 'radio', or 'checkbox'
    """
    platform = detect_platform()
    
    if platform == "google_forms":
        # Google Forms: Look for text input, select, radio buttons, or checkboxes
        
        # 1. Try text input field
        text_selectors = [
            "input[type='text'][aria-label]",
            "textarea[aria-label]",
            "input[type='text'][role='textbox']",
            "textarea[role='textbox']",
            "div[role='textbox'][contenteditable='true']",
        ]
        
        for selector in text_selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    return (element, 'text')
            except:
                pass
        
        # 2. Try select dropdown
        select_selectors = [
            "select[aria-label]",
            "div[role='listbox']",
            "div[class*='dropdown']",
        ]
        
        for selector in select_selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    return (element, 'select')
            except:
                pass
        
        # 3. Try radio buttons
        radio_selectors = [
            "input[type='radio']",
            "div[role='radio']",
            "div[class*='radio']",
        ]
        
        for selector in radio_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        return (elements, 'radio')
            except:
                pass
        
        # 4. Try checkboxes
        checkbox_selectors = [
            "input[type='checkbox']",
            "div[role='checkbox']",
            "div[class*='checkbox']",
        ]
        
        for selector in checkbox_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elements:
                    if elem.is_displayed():
                        return (elements, 'checkbox')
            except:
                pass
    
    else:  # Yaklass
        # Original Yaklass selectors
        selectors = [
            "input.gxs-answer-text-short",
            "input.gxs-answer-input",
            "input[type='text'].answer",
            "textarea.gxs-answer",
            "input[placeholder*='ответ']",
            "textarea[placeholder*='ответ']",
            ".answer-input input",
            ".answer-input textarea",
        ]
        
        for selector in selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                if element.is_displayed():
                    return (element, 'text')
            except:
                pass
    
    return (None, None)

def find_submit_button():
    """
    Find the submit button - works for both Yaklass and Google Forms.
    Yaklass: 'Ответить!' or 'сохранить'
    Google Forms: 'Submit' or 'Next'
    """
    platform = detect_platform()
    
    if platform == "google_forms":
        # Google Forms submit buttons
        selectors = [
            "button[aria-label*='Submit']",
            "button[aria-label*='submit']",
            "div[role='button'][aria-label*='Submit']",
            "//button[contains(text(), 'Submit')]",
            "//button[contains(text(), 'Next')]",
            "//button[contains(text(), 'next')]",
        ]
    else:
        # Yaklass submit buttons
        selectors = [
            "//button[contains(text(), 'Ответить!')]",
            "//button[contains(text(), 'Ответить')]",
            "//button[contains(text(), 'ответить')]",
            "//button[contains(text(), 'сохранить')]",
        ]
    
    for selector in selectors:
        try:
            if "//" in selector:
                button = driver.find_element(By.XPATH, selector)
            else:
                button = driver.find_element(By.CSS_SELECTOR, selector)
            if button.is_displayed():
                return button
        except:
            pass
    
    # Fallback: look for any visible button with submit/next/answer text
    try:
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            text = btn.text.lower()
            if any(kw in text for kw in ["ответ", "сохран", "submit", "next", "дальше"]):
                if btn.is_displayed():
                    return btn
    except:
        pass
    
    return None

def find_next_button():
    """
    Find the next button to navigate to the next question.
    Yaklass: 'Дальше'
    Google Forms: 'Next' button after form submission
    """
    platform = detect_platform()
    
    if platform == "google_forms":
        # Google Forms next buttons
        selectors = [
            "button[aria-label*='Next']",
            "//button[contains(text(), 'Next')]",
            "div[role='button'][aria-label*='Next']",
            "a[aria-label*='Next']",
        ]
    else:
        # Yaklass next buttons
        selectors = [
            "//button[contains(text(), 'Дальше')]",
            "//a[contains(text(), 'Дальше')]",
            "//button[contains(text(), 'дальше')]",
            "//a[contains(text(), 'дальше')]",
            "a[href*='next']",
            "a.next-question",
            ".pagination a.next",
        ]
    
    for selector in selectors:
        try:
            if "//" in selector:
                button = driver.find_element(By.XPATH, selector)
            else:
                button = driver.find_element(By.CSS_SELECTOR, selector)
            
            if button.is_displayed():
                return button
        except:
            pass
    
    return None

def solve_task():
    global error_log, stats
    error_log = []
    question_start_time = time.time()
    
    # Detect platform
    platform = detect_platform()
    stats["questions_solved"] += 1
    q_num = stats["questions_solved"]
    
    print(f"\n{Fore.CYAN}{'─'*50}")
    print(f"{Fore.CYAN}[?] Question #{q_num} ({platform.upper()})")
    print(f"{Fore.CYAN}{'─'*50}")
    
    full_content = ""
    
    try:
        if platform == "google_forms":
            # Google Forms: Look for question text in specific containers
            selectors = [
                "div[data-item-id] div[class*='title']",  # Question title
                "div[role='heading']",
                "div[class*='question']",
                "span[class*='title']",
                "h2[class*='heading']",
                "div[role='img'][aria-label]",  # For image-based questions
            ]
        else:
            # Yaklass: Original selectors
            selectors = [
                "div#taskhtml",
                "div.gxst-ibody",
                "div.task-body",
                "div.question-text",
                "div[class*='task']",
                "div[class*='question']",
            ]
        
        for selector in selectors:
            try:
                task_element = driver.find_element(By.CSS_SELECTOR, selector)
                if task_element.text.strip():
                    full_content = task_element.text
                    break
            except:
                pass

        if not full_content:
            try:
                body_text = driver.find_element(By.TAG_NAME, "body").text
                # Get first 500 chars of body
                full_content = body_text[:500]
            except:
                pass
        
        if full_content:
            print(f"{Fore.CYAN}[+] Extracted: {full_content[:MAX_QUESTION_PREVIEW_LENGTH]}...")
        
    except Exception as e:
        error_log.append(f"Scraping: {str(e)[:MAX_ERROR_MSG_LENGTH]}")
        print(f"{Fore.RED}[!] Failed to extract question")
        try:
            full_content = pyperclip.paste()
            if full_content:
                print(f"{Fore.CYAN}[+] Used clipboard: {full_content[:MAX_QUESTION_PREVIEW_LENGTH]}...")
        except Exception:
            print(f"{Fore.RED}[!] No question text found")
            stats["questions_failed"] += 1
            return

    if not full_content.strip():
        print(f"{Fore.RED}[!] Question is empty!")
        stats["questions_failed"] += 1
        return

    try:
        # Get answers from all models
        answers = get_answers_from_models(full_content)
        answer = verify_and_select_answer(answers)
        
        if answer is None:
            print(f"{Fore.RED}[!] No consensus reached")
            stats["questions_failed"] += 1
            try:
                now = datetime.now()
                log_path = LOG_DIR / f"failure_{now.strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'platform': platform,
                        'question': full_content[:MAX_LOGGED_CONTENT_LENGTH],
                        'responses': answers,
                        'errors': error_log,
                        'selected': None,
                        'timestamp': now.isoformat()
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return
            
    except Exception as e:
        error_log.append(f"AI Query: {str(e)[:MAX_ERROR_MSG_LENGTH]}")
        print(f"{Fore.RED}[!] AI Error: {str(e)[:60]}")
        stats["questions_failed"] += 1
        return

    try:
        success = type_answer(answer)
        if not success:
            print(f"{Fore.RED}[!] Failed to answer. Copied to clipboard.")
            pyperclip.copy(answer)
            stats["questions_failed"] += 1
            try:
                now = datetime.now()
                log_path = LOG_DIR / f"failure_typing_{now.strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'platform': platform,
                        'question': full_content[:MAX_LOGGED_CONTENT_LENGTH],
                        'answer': answer,
                        'error': 'Failed to type/select',
                        'errors': error_log,
                        'timestamp': now.isoformat()
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return
        print(f"{Fore.GREEN}[+] Answer entered: '{answer}'")
        
    except Exception as e:
        error_log.append(f"Typing: {str(e)[:MAX_ERROR_MSG_LENGTH]}")
        print(f"{Fore.RED}[!] Typing error: {str(e)[:60]}")
        stats["questions_failed"] += 1
        return
    
    try:
        time.sleep(PAGE_LOAD_DELAY)
        submit_button = find_submit_button()
        
        if submit_button:
            submit_button.click()
            print(f"{Fore.GREEN}[✓] Submitted!")
            time.sleep(SUBMIT_WAIT_TIME)
            
            # Log success
            try:
                now = datetime.now()
                elapsed = time.time() - question_start_time
                log_path = LOG_DIR / f"success_{now.strftime('%Y%m%d_%H%M%S')}.json"
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'question_num': q_num,
                        'platform': platform,
                        'question': full_content[:MAX_LOGGED_CONTENT_LENGTH],
                        'answer': answer,
                        'time_taken': round(elapsed, 2),
                        'timestamp': now.isoformat()
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        else:
            print(f"{Fore.YELLOW}[!] Submit button not found")
            stats["questions_failed"] += 1
            return
        
    except Exception as e:
        error_log.append(f"Submit: {str(e)[:MAX_ERROR_MSG_LENGTH]}")
        print(f"{Fore.RED}[!] Submit error: {str(e)[:60]}")
        stats["questions_failed"] += 1
        return
    
    # Look for next button or check if form is complete
    try:
        time.sleep(NAVIGATION_WAIT_TIME)
        next_button = find_next_button()
        
        if next_button:
            print(f"{Fore.YELLOW}[→] Moving to next question...")
            time.sleep(NAVIGATION_WAIT_TIME)
            next_button.click()
            time.sleep(NEXT_PAGE_WAIT_TIME)
            # RECURSIVE: Automatically solve next question
            solve_task()
        else:
            elapsed = time.time() - stats["start_time"]
            print(f"\n{Fore.GREEN}{'='*50}")
            print(f"{Fore.GREEN}[✓] Test completed!")
            print(f"{Fore.CYAN}    Platform: {platform.upper()}")
            print(f"{Fore.CYAN}    Solved: {stats['questions_solved']}")
            print(f"{Fore.CYAN}    Failed: {stats['questions_failed']}")
            print(f"{Fore.CYAN}    Time: {int(elapsed)}s")
            print(f"{Fore.GREEN}{'='*50}\n")
    
    except Exception as e:
        error_log.append(f"Navigation: {str(e)[:MAX_ERROR_MSG_LENGTH]}")
        print(f"{Fore.YELLOW}[!] Navigation error: {str(e)[:60]}")

print(f"{Fore.MAGENTA}=============================================")
print(f"{Fore.MAGENTA}   MULTI-PLATFORM HOMEWORK SOLVER BOT 🚀")
print(f"{Fore.MAGENTA}   ✓ Yaklass & Google Forms Support")
print(f"{Fore.MAGENTA}   ✓ Multi-AI Analysis (Perplexity + Groq)")
print(f"{Fore.MAGENTA}   ✓ Smart Answer Detection (Text/Select)")
print(f"{Fore.MAGENTA}   ✓ Auto-Submit & Auto-Navigate")
print(f"{Fore.MAGENTA}   Press {HOTKEY} to start solving!")
print(f"{Fore.MAGENTA}=============================================\n")
print(f"{Fore.CYAN}Creator: Buyndelger (caelith) — 16 years old, Mongolia")
print(f"{Fore.CYAN}Repo: created-by-caelith")

# Discover all working models on startup
print(f"\n{Fore.MAGENTA}[*] Initializing bot...")
discover_and_validate_models()

# Initialize statistics at startup
stats["start_time"] = time.time()

print(f"\n{Fore.GREEN}{'='*50}")
print(f"{Fore.GREEN}[+] Bot is ready to solve!")
print(f"{Fore.CYAN}[*] Press F8 on any question (Yaklass or Google Forms)")
print(f"{Fore.CYAN}[*] Bot auto-detects platform and answer type (text/multiple choice)")
print(f"{Fore.CYAN}[*] Bot will automatically loop through all questions")
print(f"{Fore.GREEN}{'='*50}\n")

def cleanup():
    """Gracefully shutdown the bot."""
    try:
        print(f"\n{Fore.YELLOW}[*] Shutting down gracefully...")
        if driver:
            driver.quit()
        print(f"{Fore.GREEN}[+] Goodbye! 👋\n")
    except Exception as e:
        print(f"{Fore.RED}[!] Error during shutdown: {str(e)[:60]}")

import atexit
atexit.register(cleanup)

keyboard.add_hotkey(HOTKEY, solve_task)

try:
    keyboard.wait()
except KeyboardInterrupt:
    cleanup()
except Exception as e:
    print(f"{Fore.RED}[!] Unexpected error: {str(e)[:80]}")
    cleanup()


