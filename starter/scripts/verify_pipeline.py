"""Run invariant checks; optional --database uses an isolated synthetic test namespace."""
import argparse
import ast
from pathlib import Path
import os
import sys
import uuid

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from dotenv import load_dotenv
from src.models import DocumentPage
from src.chunking import fixed_size, by_paragraph
from src.ingestion import normalize_text
from src.evaluation import evaluate_query, summarize

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--database', action='store_true')
args = parser.parse_args()
load_dotenv(root / '.env')
page = DocumentPage('synthetic-test', 'test://fixture', 'unit',
                    'A' * 120 + '\n\n' + 'B' * 30 + '\n\n' + 'C' * 230,
                    {'category': 'electronics'})
for chunks in (fixed_size(page, 80, 15), by_paragraph(page, 80)):
    covered = set()
    assert all(0 < len(c.text) <= 80 for c in chunks)
    for chunk in chunks:
        assert chunk.text == page.text[chunk.metadata['start']:chunk.metadata['end']]
        assert chunk.metadata['category'] == 'electronics'
        covered.update(range(chunk.metadata['start'], chunk.metadata['end']))
    assert all(i in covered for i, char in enumerate(page.text) if not char.isspace())
assert fixed_size(page, 80, 15) == fixed_size(page, 80, 15)
for size, overlap in ((0, 0), (10, 10), (10, -1)):
    try:
        fixed_size(page, size, overlap)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid chunk parameters accepted')
assert normalize_text('a\r\n\r\n\r\nb\u00a0c') == 'a\n\nb c'
q = {'id': 'fixture', 'split': 'dev', 'gold_source_id': 'synthetic-test',
     'gold_page': 'unit', 'evidence_quote': 'full evidence'}
hit = {'document_id': 'synthetic-test', 'page': 'unit', 'text': 'full evidence', 'metadata': {}}
trace = evaluate_query(q, [hit])
assert trace['hit_at_1'] and trace['evidence_hit_at_3']
assert not evaluate_query(q, [{**hit, 'page': 'wrong'}])['hit_at_1']
assert not evaluate_query(q, [{**hit, 'text': 'full'}])['evidence_hit_at_3']
assert evaluate_query({**q, 'alternative_gold': [{**q, 'gold_page': 'other'}]},
                      [{**hit, 'page': 'other'}])['evidence_hit_at_3']
trace['strategy'] = 'A'
assert summarize([trace])['A/dev']['hit_at_1_count'] == 1
expected = {'chunking.py': {'fixed_size', 'by_paragraph'},
            'ingestion.py': {'load_pdf', 'normalize_text'},
            'embeddings.py': {'__init__', 'embed_documents', 'embed_query'},
            'store.py': {'__init__', 'create_schema', 'upsert', 'search'},
            'evaluation.py': {'evaluate_query', 'summarize'}, 'run.py': {'main'}, 'models.py': set()}
for filename, names in expected.items():
    tree = ast.parse((root / 'src' / filename).read_text(encoding='utf-8-sig'))
    assert {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} == names
print('PASS: chunk bounds/coverage/IDs, validation, normalization, evidence, metrics, no new def names')

if args.database:
    import psycopg
    from src.models import Chunk
    from src.store import PgVectorStore
    store = PgVectorStore(os.environ['DATABASE_URL'], 3)
    store.create_schema()
    namespace = 'synthetic-unit-test-' + uuid.uuid4().hex
    a = Chunk('a', 'synthetic-test', 'test://fixture', 'p', 'alpha', {'category': 'electronics'})
    b = Chunk('b', 'synthetic-test', 'test://fixture', 'p', 'beta', {'category': 'programming'})
    try:
        store.upsert(namespace, [a, b], [[1, 0, 0], [0, 1, 0]])
        store.upsert(namespace, [a, b], [[1, 0, 0], [0, 1, 0]])
        hits = store.search(namespace, [1, 0, 0], 3)
        assert len(hits) == 2 and hits[0]['chunk_id'] == 'a' and abs(hits[0]['score'] - 1) < 1e-6
        filtered = store.search(namespace, [1, 0, 0], 3, {'category': 'programming'})
        assert len(filtered) == 1 and filtered[0]['chunk_id'] == 'b'
        store.upsert(namespace, [b], [[0, 1, 0]])
        assert [h['chunk_id'] for h in store.search(namespace, [1, 0, 0])] == ['b']
        for vector, filters in (([1, 0], None), ([0, 0, 0], None), ([float('nan'), 0, 0], None),
                                ([1, 0, 0], {'source': 'test://fixture'})):
            try:
                store.search(namespace, vector, filters=filters)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid search input accepted')
        print('PASS: actual PostgreSQL persistence/idempotence, stale removal, cosine ranking, metadata filtering, invalid inputs')
    finally:
        with psycopg.connect(os.environ['DATABASE_URL']) as connection:
            connection.execute(f'DELETE FROM {store.table} WHERE strategy=%s', (namespace,))
