"""Conservative exact evidence checks; manual labels stay separate."""
import re
from typing import Any
from .ingestion import normalize_text


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    matches = []
    for hit in hits[:3]:
        matched = False
        text = re.sub(r'\s+', ' ', normalize_text(hit['text']))
        for gold in [question] + question.get('alternative_gold', []):
            evidence = re.sub(r'\s+', ' ', normalize_text(gold['evidence_quote']))
            if evidence and hit['document_id'] == gold['gold_source_id'] and str(hit['page']) == str(gold['gold_page']) and evidence in text:
                matched = True
        matches.append(matched)
    return {'question': question, 'split': question['split'], 'hits': hits[:3],
            'evidence_matches': matches, 'hit_at_1': bool(matches and matches[0]),
            'evidence_hit_at_3': any(matches), 'human_grounding': None,
            'automatic_check': 'normalized exact match; review semantic alternatives manually'}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for trace in traces:
        key = f"{trace['strategy']}/{trace['split']}"
        group = result.setdefault(key, {'count': 0, 'hit_at_1_count': 0, 'evidence_hit_at_3_count': 0,
                                        'filtered_count': 0, 'filter_compliant_count': 0})
        group['count'] += 1
        group['hit_at_1_count'] += int(trace['hit_at_1'])
        group['evidence_hit_at_3_count'] += int(trace['evidence_hit_at_3'])
        if trace.get('applied_filter'):
            group['filtered_count'] += 1
            group['filter_compliant_count'] += int(bool(trace['hits']) and all(
                all(hit['metadata'].get(k) == v for k, v in trace['applied_filter'].items()) for hit in trace['hits']))
    for group in result.values():
        group['hit_at_1'] = group['hit_at_1_count'] / group['count']
        group['evidence_hit_at_3'] = group['evidence_hit_at_3_count'] / group['count']
        group['filter_compliance'] = group['filter_compliant_count'] / group['filtered_count'] if group['filtered_count'] else None
    return result
