"""Deterministic chunking with source offsets."""
import hashlib
import re
from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require size > 0 and 0 <= overlap < size')
    result = []
    for start in range(0, len(page.text), size - overlap):
        end = min(start + size, len(page.text))
        text = page.text[start:end]
        if text.strip():
            identity = f'{page.document_id}|{page.page}|fixed:{size}:{overlap}|{start}|{text}'
            result.append(Chunk(hashlib.sha256(identity.encode()).hexdigest(), page.document_id,
                                page.source, page.page, text, {**page.metadata, 'start': start, 'end': end}))
        if end == len(page.text):
            break
    return result


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size <= 0:
        raise ValueError('max_size must be positive')
    spans = []
    start = None
    end = 0
    for paragraph in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)', page.text):
        left, right = paragraph.span()
        if start is not None and right - start > max_size:
            spans.append((start, end))
            start = None
        while right - left > max_size:
            cut = page.text.rfind(' ', left + max_size // 2, left + max_size + 1)
            if cut <= left:
                cut = left + max_size
            spans.append((left, cut))
            left = cut
            while left < right and page.text[left].isspace():
                left += 1
        if left < right:
            if start is None:
                start = left
            end = right
    if start is not None:
        spans.append((start, end))
    result = []
    for start, end in spans:
        text = page.text[start:end]
        identity = f'{page.document_id}|{page.page}|paragraph:{max_size}|{start}|{text}'
        result.append(Chunk(hashlib.sha256(identity.encode()).hexdigest(), page.document_id,
                            page.source, page.page, text, {**page.metadata, 'start': start, 'end': end}))
    return result
