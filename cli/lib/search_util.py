from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
CASH_PATH=PROJECT_ROOT /"cash"
BM25_K1 = 1.5
BM25_B = 0.75
def load_movies()->list[dict]:
    with open(DATA_PATH, 'r') as f:
        return json.load(f)["movies"]
    
def load_stop_words()->set[str]:
    with open(PROJECT_ROOT / "data" / "stop_words.txt", 'r') as f: 
        return set(line.strip() for line in f)        