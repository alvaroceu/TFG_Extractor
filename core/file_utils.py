from urllib import request
from boilerpy3 import extractors
import json
from typing import List, Dict, Any

def read_raw_text(filepath: str) -> str:
    """Reads and returns raw text content from a local file or a URL"""
    try:
        if filepath.startswith('http://') or filepath.startswith('https://'):
            
            with request.urlopen(filepath) as response:
                extractor = extractors.ArticleExtractor()
                html = response.read().decode('utf-8')
                return extractor.get_content(html)
        else:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")

def read_databases_json(filepath: str) -> List[Dict[str, Any]]:
    """Parse dataset in json format"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {filepath}: {e}")
        return []