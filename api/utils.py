import re
import logging
from datetime import datetime

def sanitize_input(text):
    return re.sub(r"[^a-zA-Z0-9\s\.\,\?\!\-]", "", text).strip()

def clean_output(text):
    return text.replace('\n', ' ').strip()

def log_event(event_type, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.basicConfig(filename="chatbot.log", level=logging.INFO)
    logging.info(f"[{timestamp}] [{event_type.upper()}] {content}")
