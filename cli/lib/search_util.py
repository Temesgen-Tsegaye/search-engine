from pathlib import Path
import json

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_PATH=PROJECT_ROOT + '/data/movies.json'

def load_movies()->list[dict]:
    with open(DATA_PATH,'r') as f:
        return json.load(f)["movies"]