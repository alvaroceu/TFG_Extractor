from urllib import request
from boilerpy3 import extractors

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