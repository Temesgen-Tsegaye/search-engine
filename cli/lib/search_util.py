from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
CASH_PATH=PROJECT_ROOT /"cash"
def load_movies()->list[dict]:
    with open(DATA_PATH, 'r') as f:
        return json.load(f)["movies"]
    
def load_stop_words()->set[str]:
    with open(PROJECT_ROOT / "data" / "stop_words.txt", 'r') as f: 
        return set(line.strip() for line in f)        