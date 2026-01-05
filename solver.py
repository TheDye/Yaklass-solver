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

# ==================== PLATFORM DETECTION & CALIBRATION ====================

def extract_question_structure():
    """
    PRECISION EXTRACTION: Finds exact question + options in one operation.
    Algorithm:
    1. Scan ALL divs for ones with text + input fields together
    2. Score by self-containment (question text + options nearby)
    3. Return the best match with all metadata needed
    """
    print(f"\n{Fore.CYAN}[🎯] EXTRACTING QUESTION STRUCTURE (PRECISION MODE)...")
    
    try:
        all_divs = driver.find_elements(By.TAG_NAME, "div")
        print(f"    Scanning {len(all_divs)} divs...")
        
        candidates = []
        
        # Find all divs that are complete question containers
        for div in all_divs:
            try:
                if not div.is_displayed():
                    continue
                
                # Get div content
                div_text = div.text.strip()
                if not div_text or len(div_text) < 15 or len(div_text) > 500:
                    continue
                
                # Check for input elements INSIDE this div
                radios = div.find_elements(By.CSS_SELECTOR, "input[type='radio']")
                checkboxes = div.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                text_inputs = div.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea")
                selects = div.find_elements(By.CSS_SELECTOR, "select")
                
                total_inputs = len(radios) + len(checkboxes) + len(text_inputs) + len(selects)
                
                # Only consider divs with at least 1 input
                if total_inputs == 0:
                    continue
                
                # Calculate score (prefer divs with more inputs = more self-contained)
                score = len(div_text) + (total_inputs * 20)
                
                candidates.append({
                    "element": div,
                    "text": div_text,
                    "radios": radios,
                    "checkboxes": checkboxes,
                    "text_inputs": text_inputs,
                    "selects": selects,
                    "total_inputs": total_inputs,
                    "score": score
                })
            except:
                pass
        
        if not candidates:
            print(f"    {Fore.RED}✗ No question containers found")
            return None
        
        # Sort by score - highest score = most self-contained
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        
        print(f"    {Fore.GREEN}✓ Found best container (score: {best['score']})")
        print(f"    Text: {best['text'][:100]}...")
        print(f"    Inputs: {best['total_inputs']} field(s)")
        
        # Build result based on what type of input we found
        result = {
            "question_text": best["text"],
            "question_element": best["element"],
            "options": [],
            "field_type": None,
            "is_multiple_choice": False
        }
        
        # EXTRACT OPTIONS based on field type
        if len(best["radios"]) > 0:
            result["field_type"] = "radio"
            result["is_multiple_choice"] = True
            print(f"    {Fore.GREEN}✓ Field: RADIO ({len(best['radios'])} options)")
            
            for radio in best["radios"]:
                try:
                    if not radio.is_displayed():
                        continue
                    
                    label_text = _extract_option_label(radio)
                    if label_text:
                        result["options"].append({
                            "text": label_text,
                            "element": radio,
                            "type": "radio"
                        })
                except:
                    pass
        
        elif len(best["checkboxes"]) > 0:
            result["field_type"] = "checkbox"
            result["is_multiple_choice"] = True
            print(f"    {Fore.GREEN}✓ Field: CHECKBOX ({len(best['checkboxes'])} options)")
            
            for checkbox in best["checkboxes"]:
                try:
                    if not checkbox.is_displayed():
                        continue
                    
                    label_text = _extract_option_label(checkbox)
                    if label_text:
                        result["options"].append({
                            "text": label_text,
                            "element": checkbox,
                            "type": "checkbox"
                        })
                except:
                    pass
        
        elif len(best["selects"]) > 0:
            result["field_type"] = "select"
            print(f"    {Fore.GREEN}✓ Field: SELECT")
            
            select_elem = best["selects"][0]
            try:
                from selenium.webdriver.support.select import Select
                select = Select(select_elem)
                for option in select.options:
                    opt_text = option.text.strip()
                    if opt_text:
                        result["options"].append({
                            "text": opt_text,
                            "element": option,
                            "type": "select"
                        })
            except:
                pass
        
        elif len(best["text_inputs"]) > 0:
            result["field_type"] = "text"
            print(f"    {Fore.GREEN}✓ Field: TEXT")
            result["options"] = [{
                "text": "text_input",
                "element": best["text_inputs"][0],
                "type": "text"
            }]
        
        if result["options"]:
            print(f"    {Fore.CYAN}    Found {len(result['options'])} option(s)")
        
        return result
    
    except Exception as e:
        print(f"    {Fore.RED}✗ Error: {str(e)[:80]}")
        error_log.append(f"Question extraction: {str(e)[:80]}")
        return None

def _extract_option_label(input_element):
    """
    Extract label text for a single input element (radio/checkbox).
    Tries multiple strategies in order of reliability.
    """
    label_text = ""
    
    # Strategy 1: Parent label tag
    try:
        parent_label = input_element.find_element(By.XPATH, "ancestor::label[1]")
        label_text = parent_label.text.strip()
        if label_text:
            return label_text
    except:
        pass
    
    # Strategy 2: aria-label attribute
    label_text = input_element.get_attribute("aria-label") or ""
    if label_text:
        return label_text.strip()
    
    # Strategy 3: Following sibling span
    try:
        sibling = input_element.find_element(By.XPATH, "following-sibling::span[1]")
        label_text = sibling.text.strip()
        if label_text:
            return label_text
    except:
        pass
    
    # Strategy 4: Parent div text (most likely to have unwanted text)
    try:
        parent = input_element.find_element(By.XPATH, "ancestor::div[1]")
        label_text = parent.text.strip()
        if label_text:
            return label_text
    except:
        pass
    
    # Strategy 5: value attribute
    label_text = input_element.get_attribute("value") or ""
    if label_text:
        return label_text.strip()
    
    return ""

def match_answer_to_option(ai_answer, available_options):
    """
    BULLETPROOF MATCHING: Match AI answer to exact option with 4-strategy fallback.
    """
    print(f"\n{Fore.CYAN}[🎯] MATCHING ANSWER TO OPTIONS...")
    print(f"    AI Answer: '{ai_answer}'")
    print(f"    Options: {len(available_options)}")
    
    if not available_options:
        return None
    
    # Normalize the AI answer once
    ai_norm = normalize_answer(ai_answer)
    
    best_match = None
    best_score = 0.0
    
    # Evaluate each option
    for option in available_options:
        opt_text = option["text"]
        opt_norm = normalize_answer(opt_text)
        
        # Strategy 1: EXACT MATCH
        if ai_norm == opt_norm:
            print(f"    {Fore.GREEN}✓✓✓ EXACT MATCH: '{opt_text}'")
            return {"option": option, "score": 1.0}
        
        # Strategy 2: SUBSTRING (AI answer is part of option text)
        if ai_norm in opt_norm or opt_norm in ai_norm:
            score = 0.95
            print(f"    {Fore.GREEN}✓✓ Substring match: '{opt_text}' ({score:.0%})")
            if score > best_score:
                best_score = score
                best_match = {"option": option, "score": score}
            continue
        
        # Strategy 3: WORD OVERLAP (Jaccard similarity)
        ai_words = set(ai_norm.split())
        opt_words = set(opt_norm.split())
        
        if ai_words and opt_words:
            overlap = len(ai_words & opt_words) / max(len(ai_words), len(opt_words))
            
            if overlap >= 0.6:
                print(f"    {Fore.YELLOW}↳ Word overlap: '{opt_text}' ({overlap:.0%})")
                if overlap > best_score:
                    best_score = overlap
                    best_match = {"option": option, "score": overlap}
                continue
        
        # Strategy 4: FIRST WORD MATCH
        ai_first = ai_norm.split()[0] if ai_norm else ""
        opt_first = opt_norm.split()[0] if opt_norm else ""
        
        if ai_first and opt_first and ai_first == opt_first:
            score = 0.5
            print(f"    {Fore.YELLOW}↳ First word: '{opt_text}' ({score:.0%})")
            if score > best_score:
                best_score = score
                best_match = {"option": option, "score": score}
    
    # Return best match if found
    if best_match and best_score >= 0.5:
        print(f"\n    {Fore.GREEN}[✓] MATCH SELECTED")
        print(f"    Option: '{best_match['option']['text']}'")
        print(f"    Confidence: {best_match['score']:.0%}")
        return best_match
    
    # Fallback: No match found
    print(f"\n    {Fore.RED}[!] No match found (best score: {best_score:.0%})")
    return None

def select_answer_option(matched_option):
    """
    RELIABLE SELECTION: Select the matched option with 4 fallback strategies.
    """
    print(f"\n{Fore.CYAN}[✓] SELECTING OPTION...")
    
    try:
        option_elem = matched_option["option"]["element"]
        field_type = matched_option["option"]["type"]
        
        if field_type in ["radio", "checkbox"]:
            # Try 4 strategies in order of reliability
            strategies = [
                ("Direct click", lambda: option_elem.click()),
                ("Parent label click", lambda: option_elem.find_element(By.XPATH, "ancestor::label[1]").click()),
                ("Scroll + click", lambda: (driver.execute_script("arguments[0].scrollIntoView(true);", option_elem), time.sleep(0.3), option_elem.click())),
                ("JavaScript click", lambda: driver.execute_script("arguments[0].click();", option_elem))
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    print(f"    Trying: {strategy_name}...")
                    strategy_func()
                    time.sleep(0.5)
                    
                    # Verify selection
                    if option_elem.is_selected():
                        print(f"    {Fore.GREEN}✓ Success!")
                        return True
                except Exception as e:
                    print(f"    {Fore.YELLOW}  Failed: {str(e)[:40]}")
                    continue
        
        elif field_type == "select":
            try:
                from selenium.webdriver.support.select import Select
                select_elem = option_elem.find_element(By.XPATH, "ancestor::select[1]")
                select = Select(select_elem)
                select.select_by_value(option_elem.get_attribute("value"))
                time.sleep(0.3)
                print(f"    {Fore.GREEN}✓ Selected from dropdown")
                return True
            except Exception as e:
                print(f"    {Fore.RED}✗ Dropdown failed: {str(e)[:40]}")
        
        elif field_type == "text":
            try:
                option_elem.click()
                time.sleep(0.2)
                print(f"    {Fore.GREEN}✓ Text field ready")
                return True
            except Exception as e:
                print(f"    {Fore.RED}✗ Text field failed: {str(e)[:40]}")
        
        print(f"    {Fore.RED}✗ All strategies failed")
        return False
    
    except Exception as e:
        print(f"    {Fore.RED}[!] Error: {str(e)[:80]}")
        error_log.append(f"Select error: {str(e)[:80]}")
        return False

def auto_calibrate_page():
    """
    Deprecated: Use extract_question_structure() instead.
    This function is kept for backward compatibility but does nothing.
    """
    print(f"\n{Fore.YELLOW}[!] auto_calibrate_page() is deprecated. Using extract_question_structure() instead.")
    return None

def calibrate_google_forms():
    """
    Calibrate Google Forms page structure.
    Analyzes the page to find questions, answer fields, and their types.
    Returns calibration data for current page.
    """
    try:
        calibration = {
            "question_elements": [],
            "answer_field_info": None,
            "field_type": None,
            "has_multiple_questions": False,
            "page_structure": {}
        }
        
        # Find all potential question containers
        question_selectors = [
            ("div[data-item-id]", "data-item-id"),
            ("div[role='heading']", "role='heading'"),
            ("div[class*='question']", "class*='question'"),
            ("div[class*='item']", "class*='item'"),
            ("div[class*='prompt']", "class*='prompt'"),
        ]
        
        all_questions = []
        for selector, desc in question_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    all_questions.extend([(elem, desc) for elem in elements if elem.is_displayed()])
            except:
                pass
        
        # Deduplicate and store
        calibration["question_elements"] = list(set(all_questions)) if all_questions else []
        
        # Find answer field and determine type
        text_field = find_google_forms_text_field()
        if text_field:
            calibration["answer_field_info"] = text_field
            calibration["field_type"] = "text"
            print(f"{Fore.CYAN}[+] Detected field type: TEXT INPUT")
        else:
            # Check for multiple choice
            radio_field = find_google_forms_radio_buttons()
            if radio_field:
                calibration["answer_field_info"] = radio_field
                calibration["field_type"] = "radio"
                print(f"{Fore.CYAN}[+] Detected field type: RADIO BUTTONS ({radio_field.get('count', 0)} options)")
            else:
                checkbox_field = find_google_forms_checkboxes()
                if checkbox_field:
                    calibration["answer_field_info"] = checkbox_field
                    calibration["field_type"] = "checkbox"
                    print(f"{Fore.CYAN}[+] Detected field type: CHECKBOXES ({checkbox_field.get('count', 0)} options)")
                else:
                    select_field = find_google_forms_select()
                    if select_field:
                        calibration["answer_field_info"] = select_field
                        calibration["field_type"] = "select"
                        print(f"{Fore.CYAN}[+] Detected field type: SELECT DROPDOWN")
        
        return calibration
    except Exception as e:
        error_log.append(f"Calibration error: {str(e)[:80]}")
        return None

def find_google_forms_text_field():
    """
    Find text input/textarea field in Google Forms.
    Returns element with additional metadata.
    """
    text_selectors = [
        "input[type='text'][aria-label]",
        "textarea[aria-label]",
        "input[type='text'][aria-describedby]",
        "textarea[aria-describedby]",
        "input[type='text'][class*='input']",
        "textarea[class*='textarea']",
        "input[type='email']",
        "input[type='url']",
        "input[type='number']",
        "div[role='textbox'][contenteditable='true']",
        "input[type='text']:not([aria-hidden])",
        "textarea:not([aria-hidden])",
    ]
    
    for selector in text_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for elem in elements:
                if elem.is_displayed() and not elem.get_attribute('aria-hidden'):
                    # Get parent context for better identification
                    parent = elem.find_element(By.XPATH, "..")
                    return {
                        "element": elem,
                        "selector": selector,
                        "parent": parent,
                        "type": "text"
                    }
        except:
            pass
    
    return None

def find_google_forms_radio_buttons():
    """
    Find radio button group in Google Forms with their labels.
    Returns list of (element, label_text) tuples with metadata.
    """
    radio_selectors = [
        # Primary: Find by data-item-id (question container) then radio inputs within
        ("div[data-item-id] input[type='radio']", "data-item-id input[type='radio']"),
        # Secondary: Role-based
        ("div[role='radio']", "role='radio'"),
        # Tertiary: Find option containers with radio
        ("div[class*='option'] input[type='radio']", "option input[type='radio']"),
        # Fallback: Just radio inputs
        ("input[type='radio']", "input[type='radio']"),
    ]
    
    for selector, desc in radio_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            visible_elements = []
            
            for elem in elements:
                try:
                    if not elem.is_displayed():
                        continue
                    if elem.get_attribute('aria-hidden') == 'true':
                        continue
                    
                    # Get associated label text
                    label_text = ""
                    
                    # Try to find parent label
                    try:
                        parent_label = elem.find_element(By.XPATH, "ancestor::label[1]")
                        label_text = parent_label.text.strip()
                    except:
                        pass
                    
                    # If no label, try aria-label
                    if not label_text:
                        label_text = elem.get_attribute('aria-label') or ""
                    
                    # If still no label, try next sibling span
                    if not label_text:
                        try:
                            next_span = elem.find_element(By.XPATH, "following-sibling::span[1]")
                            label_text = next_span.text.strip()
                        except:
                            pass
                    
                    # Last resort: get parent div text
                    if not label_text:
                        try:
                            parent_div = elem.find_element(By.XPATH, "ancestor::div[1]")
                            label_text = parent_div.text.strip()
                        except:
                            label_text = elem.text.strip() or elem.get_attribute('value') or ""
                    
                    visible_elements.append((elem, label_text))
                except:
                    pass
            
            if len(visible_elements) > 1:  # Multiple options
                print(f"{Fore.CYAN}[+] Found {len(visible_elements)} radio options using: {desc}")
                return {
                    "elements": visible_elements,
                    "selector": selector,
                    "count": len(visible_elements),
                    "type": "radio"
                }
        except:
            pass
    
    return None

def find_google_forms_checkboxes():
    """
    Find checkbox group in Google Forms.
    Returns list of checkbox elements with metadata.
    """
    checkbox_selectors = [
        "input[type='checkbox']",
        "div[role='checkbox']",
        "div[class*='checkbox'][role='button']",
        "label[class*='checkbox']",
        "div[class*='option'][class*='checkbox']",
    ]
    
    for selector in checkbox_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            visible_elements = [e for e in elements if e.is_displayed() and not e.get_attribute('aria-hidden')]
            if len(visible_elements) > 1:  # Multiple options
                return {
                    "elements": visible_elements,
                    "selector": selector,
                    "count": len(visible_elements),
                    "type": "checkbox"
                }
        except:
            pass
    
    return None

def find_google_forms_select():
    """
    Find dropdown select in Google Forms.
    Returns select element with metadata.
    """
    select_selectors = [
        "select[aria-label]",
        "select[aria-describedby]",
        "div[role='listbox']",
        "div[class*='dropdown']",
        "select",
    ]
    
    for selector in select_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, selector)
            if element.is_displayed() and not element.get_attribute('aria-hidden'):
                return {
                    "element": element,
                    "selector": selector,
                    "type": "select"
                }
        except:
            pass
    
    return None

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

def find_current_question_element():
    """
    Find the first unanswered question element on the page.
    Uses ultra-smart extraction for precise identification.
    """
    platform = detect_platform()
    
    if platform == "google_forms":
        # Use ultra-smart extraction
        question_structure = extract_question_structure()
        
        if question_structure:
            return question_structure["question_element"]
        else:
            print(f"{Fore.RED}[!] Ultra-smart extraction failed")
            return None
    
    else:  # Yaklass
        yaklass_selectors = [
            "div#taskhtml",
            "div.gxst-ibody",
            "div.task-body",
        ]
        
        for selector in yaklass_selectors:
            try:
                elem = driver.find_element(By.CSS_SELECTOR, selector)
                if elem.is_displayed():
                    return elem
            except:
                pass
    
    return None

def detect_question_field_type(question_element=None):
    """Detect field type for current question."""
    if question_element is None:
        question_element = find_current_question_element()
    
    if question_element is None:
        return None
    
    try:
        # Text input
        text_inputs = question_element.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea, input[type='email'], input[type='number']")
        for elem in text_inputs:
            if elem.is_displayed() and elem.get_attribute('aria-hidden') != 'true':
                print(f"{Fore.CYAN}[+] Field type: TEXT INPUT")
                return (elem, "text")
        
        # Radio buttons
        radio_inputs = question_element.find_elements(By.CSS_SELECTOR, "input[type='radio']")
        if len(radio_inputs) > 1:
            radio_with_labels = []
            for radio in radio_inputs:
                if not radio.is_displayed():
                    continue
                
                label_text = ""
                try:
                    parent_label = radio.find_element(By.XPATH, "ancestor::label[1]")
                    label_text = parent_label.text.strip()
                except:
                    pass
                
                if not label_text:
                    label_text = radio.get_attribute('aria-label') or ""
                
                if not label_text:
                    try:
                        next_span = radio.find_element(By.XPATH, "following-sibling::span[1]")
                        label_text = next_span.text.strip()
                    except:
                        pass
                
                if not label_text:
                    try:
                        parent_div = radio.find_element(By.XPATH, "ancestor::div[1]")
                        label_text = parent_div.text.strip()
                    except:
                        label_text = radio.get_attribute('value') or ""
                
                radio_with_labels.append((radio, label_text))
            
            if radio_with_labels:
                print(f"{Fore.CYAN}[+] Field type: RADIO ({len(radio_with_labels)} options)")
                return (radio_with_labels, "radio")
        
        # Checkboxes
        checkboxes = question_element.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
        if len(checkboxes) >= 1:
            checkbox_with_labels = []
            for checkbox in checkboxes:
                if not checkbox.is_displayed():
                    continue
                
                label_text = ""
                try:
                    parent_label = checkbox.find_element(By.XPATH, "ancestor::label[1]")
                    label_text = parent_label.text.strip()
                except:
                    pass
                
                if not label_text:
                    label_text = checkbox.get_attribute('aria-label') or ""
                
                if not label_text:
                    try:
                        parent_div = checkbox.find_element(By.XPATH, "ancestor::div[1]")
                        label_text = parent_div.text.strip()
                    except:
                        label_text = checkbox.get_attribute('value') or ""
                
                checkbox_with_labels.append((checkbox, label_text))
            
            if checkbox_with_labels:
                print(f"{Fore.CYAN}[+] Field type: CHECKBOX ({len(checkbox_with_labels)} options)")
                return (checkbox_with_labels, "checkbox")
        
        # Select dropdown
        try:
            select = question_element.find_element(By.CSS_SELECTOR, "select")
            if select.is_displayed():
                print(f"{Fore.CYAN}[+] Field type: SELECT")
                return (select, "select")
        except:
            pass
    
    except Exception as e:
        error_log.append(f"Field type detection: {str(e)[:80]}")
    
    return None

def extract_question_text(platform, question_element=None):
    """Extract question text from specific question element."""
    if question_element is None:
        question_element = find_current_question_element()
    
    full_content = ""
    
    if question_element is None:
        print(f"{Fore.YELLOW}[!] Could not find question element")
        return ""
    
    try:
        if platform == "google_forms":
            question_selectors = [
                "div[role='heading']",
                "div[class*='prompt']",
                "span[class*='title']",
                "div[class*='text']",
                "span",
            ]
            
            for selector in question_selectors:
                try:
                    elements = question_element.find_elements(By.CSS_SELECTOR, selector)
                    for elem in elements:
                        text = elem.text.strip()
                        if text and len(text) > 5 and "*" not in text[:3]:
                            full_content = text
                            break
                    if full_content:
                        break
                except:
                    pass
            
            if not full_content:
                full_content = question_element.text.strip()
        
        else:
            full_content = question_element.text.strip()
    
    except Exception as e:
        error_log.append(f"Question extraction: {str(e)[:80]}")
    
    full_content = full_content.replace("*", "").strip()
    return full_content

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
        options_element: List of (element, label_text) tuples for radio/checkbox, or select element
        answer: The answer text to match
        is_radio: True if radio buttons
        is_checkbox: True if checkboxes
    
    Returns:
        bool: True if selection successful
    """
    try:
        answer_norm = normalize_answer(answer)
        print(f"{Fore.CYAN}[*] Matching answer '{answer}' to options...")
        
        if isinstance(options_element, list):
            # Radio or checkbox list with (element, label_text) tuples
            best_match = None
            best_score = 0.0
            best_label = ""
            
            for option_elem, option_text in options_element:
                try:
                    if not option_text:
                        option_text = option_elem.get_attribute('value') or ""
                    
                    if option_text:
                        score = similarity_score(answer, option_text)
                        print(f"{Fore.YELLOW}    Option '{option_text[:30]}': score={score:.2f}")
                        if score > best_score:
                            best_score = score
                            best_match = option_elem
                            best_label = option_text
                except Exception as e:
                    error_log.append(f"Option eval: {str(e)[:60]}")
            
            if best_match and best_score >= 0.3:
                print(f"{Fore.GREEN}[+] Matching option: '{best_label}' (score={best_score:.2f})")
                try:
                    best_match.click()
                    time.sleep(0.5)
                    return True
                except Exception as click_err:
                    # Try clicking parent if input is not clickable
                    try:
                        parent = best_match.find_element(By.XPATH, "ancestor::label[1]")
                        parent.click()
                        time.sleep(0.5)
                        return True
                    except:
                        # Try scrolling and clicking again
                        driver.execute_script("arguments[0].scrollIntoView(true);", best_match)
                        time.sleep(0.3)
                        best_match.click()
                        time.sleep(0.5)
                        return True
            else:
                print(f"{Fore.RED}[!] No good match found (best score: {best_score:.2f})")
        
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
    Uses calibration for Google Forms for better stability.
    Returns: (field_element, field_type) where field_type is 'text', 'select', 'radio', or 'checkbox'
    """
    platform = detect_platform()
    
    if platform == "google_forms":
        # Use calibration for Google Forms
        calibration = calibrate_google_forms()
        
        if calibration and calibration.get("answer_field_info"):
            field_info = calibration["answer_field_info"]
            field_type = calibration.get("field_type")
            
            if field_type == "text":
                return (field_info["element"], "text")
            elif field_type == "radio":
                return (field_info["elements"], "radio")
            elif field_type == "checkbox":
                return (field_info["elements"], "checkbox")
            elif field_type == "select":
                return (field_info["element"], "select")
        
        # Fallback: Manual detection without calibration
        # 1. Try text input field
        text_field = find_google_forms_text_field()
        if text_field:
            return (text_field["element"], "text")
        
        # 2. Try select dropdown
        select_field = find_google_forms_select()
        if select_field:
            return (select_field["element"], "select")
        
        # 3. Try radio buttons
        radio_field = find_google_forms_radio_buttons()
        if radio_field:
            return (radio_field["elements"], "radio")
        
        # 4. Try checkboxes
        checkbox_field = find_google_forms_checkboxes()
        if checkbox_field:
            return (checkbox_field["elements"], "checkbox")
    
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
                    return (element, "text")
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
    
    platform = detect_platform()
    stats["questions_solved"] += 1
    q_num = stats["questions_solved"]
    
    print(f"\n{Fore.CYAN}{'─'*60}")
    print(f"{Fore.CYAN}[?] Question #{q_num} ({platform.upper()})")
    print(f"{Fore.CYAN}{'─'*60}")
    
    try:
        # ULTRA-SMART WORKFLOW FOR GOOGLE FORMS
        if platform == "google_forms":
            print(f"{Fore.CYAN}[*] Starting ultra-smart question extraction...")
            
            # Step 1: Extract complete question structure (text + options)
            question_structure = extract_question_structure()
            if not question_structure:
                print(f"{Fore.RED}[!] Could not extract question structure")
                stats["questions_failed"] += 1
                return
            
            full_content = question_structure["question_text"]
            available_options = question_structure["options"]
            field_type = question_structure["field_type"]
            
            print(f"{Fore.CYAN}[+] Question text: {full_content[:MAX_QUESTION_PREVIEW_LENGTH]}...")
            print(f"{Fore.CYAN}[+] Found {len(available_options)} option(s)")
            
            if not full_content.strip():
                print(f"{Fore.RED}[!] Question text is empty!")
                stats["questions_failed"] += 1
                return
            
            # Step 2: Get answers from AI models
            print(f"{Fore.CYAN}[*] Getting AI answers...")
            answers = get_answers_from_models(full_content)
            ai_answer = verify_and_select_answer(answers)
            
            if not ai_answer:
                print(f"{Fore.RED}[!] No consensus reached from AI models")
                stats["questions_failed"] += 1
                return
            
            # Step 3: INTELLIGENT MATCHING - Match AI answer to actual options
            print(f"{Fore.CYAN}[*] Matching AI answer to available options...")
            matched = match_answer_to_option(ai_answer, available_options)
            
            if not matched:
                print(f"{Fore.RED}[!] Could not match answer to any option")
                stats["questions_failed"] += 1
                return
            
            # Step 4: PRECISE SELECTION - Select the matched option
            print(f"{Fore.CYAN}[*] Selecting the matched option...")
            success = select_answer_option(matched)
            
            if not success:
                print(f"{Fore.RED}[!] Failed to select answer option")
                stats["questions_failed"] += 1
                return
            
            print(f"{Fore.GREEN}[✓] Answer selected successfully!")
        
        else:  # YAKLASS
            print(f"{Fore.CYAN}[*] Finding current question...")
            question_element = find_current_question_element()
            
            if question_element is None:
                print(f"{Fore.RED}[!] Could not find current question element")
                stats["questions_failed"] += 1
                return
            
            print(f"{Fore.CYAN}[*] Extracting question text...")
            full_content = extract_question_text(platform, question_element)
            
            if not full_content.strip():
                print(f"{Fore.RED}[!] Question text is empty!")
                stats["questions_failed"] += 1
                return
            
            print(f"{Fore.CYAN}[+] Question: {full_content[:MAX_QUESTION_PREVIEW_LENGTH]}...")
            
            # Get answers from AI
            print(f"{Fore.CYAN}[*] Getting AI answers...")
            answers = get_answers_from_models(full_content)
            ai_answer = verify_and_select_answer(answers)
            
            if not ai_answer:
                print(f"{Fore.RED}[!] No consensus reached")
                stats["questions_failed"] += 1
                return
            
            # Type answer for Yaklass
            try:
                answer_field, field_type = find_answer_field()
                if field_type == "text" and answer_field:
                    answer_field.click()
                    time.sleep(0.2)
                    human_type(answer_field, ai_answer)
                    print(f"{Fore.GREEN}[+] Typed: '{ai_answer}'")
                else:
                    print(f"{Fore.RED}[!] Could not find text field for Yaklass")
                    stats["questions_failed"] += 1
                    return
            except Exception as e:
                print(f"{Fore.RED}[!] Failed to type answer: {str(e)[:60]}")
                stats["questions_failed"] += 1
                return
        
        # Step 5: Submit the answer
        print(f"{Fore.CYAN}[*] Submitting answer...")
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
                        'answer': ai_answer,
                        'time_taken': round(elapsed, 2),
                        'timestamp': now.isoformat()
                    }, f, ensure_ascii=False, indent=2)
            except:
                pass
        else:
            print(f"{Fore.YELLOW}[!] Submit button not found")
            stats["questions_failed"] += 1
            return
        
        # Step 6: Move to next question or finish
        print(f"{Fore.CYAN}[*] Looking for next question...")
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
            print(f"\n{Fore.GREEN}{'='*60}")
            print(f"{Fore.GREEN}[✓] TEST COMPLETED!")
            print(f"{Fore.CYAN}    Platform: {platform.upper()}")
            print(f"{Fore.CYAN}    Total Solved: {stats['questions_solved']}")
            print(f"{Fore.CYAN}    Total Failed: {stats['questions_failed']}")
            print(f"{Fore.CYAN}    Time Elapsed: {int(elapsed)}s")
            print(f"{Fore.GREEN}{'='*60}\n")
    
    except Exception as e:
        error_log.append(f"Solve error: {str(e)[:MAX_ERROR_MSG_LENGTH]}")
        print(f"{Fore.RED}[!] Unexpected error: {str(e)[:80]}")
        stats["questions_failed"] += 1
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


