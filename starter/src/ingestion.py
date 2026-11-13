"""Source extraction and conservative normalization."""
import logging
import re
import unicodedata
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    result = []
    for number, page in enumerate(PdfReader(path).pages, 1):
        text = normalize_text(page.extract_text() or '')
        if not text:
            logging.warning('No text in %s page %s; OCR required', path, number)
        result.append(DocumentPage(document_id, source, number, text,
                                   {'document_type': 'pdf', 'extraction_empty': not bool(text)}))
    return result


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text).replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\u00a0', ' ').replace('\u200b', '').replace('\ufeff', '')
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    return re.sub(r'\n{3,}', '\n\n', text).strip()
