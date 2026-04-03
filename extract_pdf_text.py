import os
import re
from pathlib import Path
from typing import List

# Configuration
PDF_PATH = 'AI_Document_Intelligence_System.pdf'
OUT_PATH = 'AI_Document_Intelligence_System.extracted.txt'

try:
    from pypdf import PdfReader
except ImportError:
    print('ERROR: pypdf not installed. Install with: pip install pypdf')
    exit(1)

def extract_from_file(path: str, out_path: str):
    print(f"Reading: {path}")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return False
        
    try:
        reader = PdfReader(path)
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ''
                # Clean artifacts that break "stitching"
                # 1. Remove standalone page numbers at start/end of lines
                t = re.sub(r'^\s*\d+\s*$', '', t, flags=re.MULTILINE)
                # 2. Remove common headers (Case-insensitive 'Page X')
                t = re.sub(r'(?i)page\s*\d+', '', t)
                # 3. Fix words split across lines (e.g., "hyphen- ation")
                t = re.sub(r'(\w+)-\n\s*(\w+)', r'\1\2', t)
            except Exception:
                t = ''
            chunks.append(f"\n\n===== PAGE {i + 1} =====\n\n{t}")
        
        Path(out_path).write_text(''.join(chunks), encoding='utf-8', errors='ignore')
        print(f'OK: Wrote {out_path}')
        return True
    except Exception as e:
        print(f"Failed to extract {path}: {e}")
        return False

# Main Execution
if __name__ == "__main__":
    if os.path.exists(PDF_PATH):
        extract_from_file(PDF_PATH, OUT_PATH)
    else:
        # Fallback: Find any PDF in the current directory
        pdfs = list(Path('.').glob('*.pdf'))
        if pdfs:
            print(f"Target {PDF_PATH} not found. Trying {pdfs[0].name} instead...")
            extract_from_file(str(pdfs[0]), OUT_PATH)
        else:
            print(f"Error: No PDF files found in {os.getcwd()}")
