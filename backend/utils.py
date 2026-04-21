import re

def clean_pdf_text(text: str) -> str:
    """Clean PDF extraction artifacts."""
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\b(BOS|EOS|FIG\.?|Figure\s*\d+)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\w{3,})\s+([a-z]{2,4})\b', 
                  lambda m: m.group(1) + m.group(2) if len(m.group(1) + m.group(2)) <= 15 else m.group(0), 
                  text)
    
    fixes = {
        r'R ises': 'Rises', r'Farewellto': 'Farewell to', r'SunAlso': 'Sun Also',
        r'distributi on': 'distribution', r'generati on': 'generation', r'informati on': 'information',
        r'conta in': 'contain', r'retriev er': 'retriever', r'generat or': 'generator',
        r'th at': 'that', r'wh ich': 'which', r'for m': 'form', r'ar Xiv': 'arXiv',
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
    lines = text.split('\n')
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: cleaned.append('\n'); continue
        if i < len(lines) - 1 and not re.search(r'[.!?]\s*$', line): cleaned.append(line + ' ')
        else: cleaned.append(line + '\n')
    text = ''.join(cleaned)
    return re.sub(r'\s{3,}', '  ', re.sub(r'\n{3,}', '\n\n', text)).strip()

def get_citation_snippet(text: str, max_len: int = 150) -> str:
    """Return a clean, meaningful snippet for citations."""
    clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    clean = re.sub(r'\b(BOS|EOS|FIG\.?|Figure\s*\d+)\b', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\s{2,}', ' ', clean)
    sentences = re.split(r'(?<=[.!?])\s+', clean.strip())
    valid = [s.strip() for s in sentences if len(s.strip()) > 10 and not re.match(r'^[\s\.\,\;\:\!\?]+$', s)]
    first = valid[0] if valid else clean[:max_len]
    if len(first) > max_len:
        first = first[:max_len].rsplit(' ', 1)[0] + "..."
    return first.strip()