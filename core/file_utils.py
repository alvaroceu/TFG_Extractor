from urllib import request

def read_raw_text(filepath: str) -> str:
    try:
        if filepath.startswith('http://') or filepath.startswith('https://'):
            
            with request.urlopen(filepath) as response:
                return response.read().decode('utf-8')
        else:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")