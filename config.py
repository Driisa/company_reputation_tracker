import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///company_tracker.db")
DB_CONNECT_ARGS = {"check_same_thread": False}

# API settings
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_CONTENT_LENGTH = 3000
REQUEST_DELAY_MIN = 0.5
REQUEST_DELAY_MAX = 1.5

# Application settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
ARTICLE_LIMIT = int(os.getenv("ARTICLE_LIMIT", "15"))
DAYS_TO_FETCH = int(os.getenv("DAYS_TO_FETCH", "7"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "logs")
DATA_DIR = os.getenv("DATA_DIR", "static/data")

# Create required directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)