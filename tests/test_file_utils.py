import unittest
import os
from core.file_utils import *

class TestFileUtils(unittest.TestCase):

    def test_read_raw_text(self):
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, "test.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("Sample text content.")
            
            content = read_raw_text(file_path)
            self.assertEqual(content, "Sample text content.")
