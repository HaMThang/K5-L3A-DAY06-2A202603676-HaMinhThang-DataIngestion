"""Transactional pgvector storage; upsert replaces a full strategy snapshot."""
import math
from typing import Any
import psycopg
from psycopg.types.json import Jsonb
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url or not isinstance(dimensions, int) or not 0 < dimensions <= 16000:
            raise ValueError('Database URL and valid vector dimensions required')
        self.database_url = database_url
        self.dimensions = dimensions
        # Separate physical columns for different dimensions; namespaces isolate models/runs.
        self.table = f'ingestion_chunks_{dimensions}'

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url, connect_timeout=10) as conn:
            conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            conn.execute(f'''CREATE TABLE IF NOT EXISTS {self.table} (
                strategy text NOT NULL, chunk_id text NOT NULL,
                document_id text NOT NULL, source text NOT NULL, page text NOT NULL,
                content text NOT NULL, metadata jsonb NOT NULL,
                embedding vector({self.dimensions}) NOT NULL,
                PRIMARY KEY (strategy, chunk_id))''')
            actual = conn.execute('''SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute WHERE attrelid = %s::regclass
                AND attname = 'embedding' ''', (self.table,)).fetchone()[0]
            if actual != f'vector({self.dimensions})':
                raise ValueError(f'Existing embedding type {actual} differs from model dimensions')
            conn.execute(f'CREATE INDEX IF NOT EXISTS {self.table}_metadata_idx ON {self.table} USING gin(metadata)')

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not strategy or not chunks or len(chunks) != len(vectors):
            raise ValueError('Require complete nonempty snapshot and matching vectors')
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError('Duplicate chunk IDs')
        rows = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector) or not any(vector):
                raise ValueError('Invalid vector dimension, values or zero vector')
            rows.append((strategy, chunk.chunk_id, chunk.document_id, chunk.source, str(chunk.page),
                         chunk.text, Jsonb(chunk.metadata), '[' + ','.join(str(float(v)) for v in vector) + ']'))
        with psycopg.connect(self.database_url, connect_timeout=10) as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (strategy,))
            conn.execute(f'DELETE FROM {self.table} WHERE strategy = %s', (strategy,))
            with conn.cursor() as cursor:
                cursor.executemany(f'''INSERT INTO {self.table}
                    (strategy,chunk_id,document_id,source,page,content,metadata,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)''', rows)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        permitted = {'category', 'section', 'language', 'version', 'document_type', 'year', 'audience'}
        if filters and (set(filters) - permitted or any(not isinstance(v, (str, int, bool)) for v in filters.values())):
            raise ValueError('Unsupported metadata filter')
        if top_k <= 0 or len(query_vector) != self.dimensions or not all(math.isfinite(v) for v in query_vector) or not any(query_vector):
            raise ValueError('Invalid query vector or top_k')
        vector = '[' + ','.join(str(float(v)) for v in query_vector) + ']'
        with psycopg.connect(self.database_url, connect_timeout=10) as conn:
            rows = conn.execute(f'''SELECT chunk_id,document_id,source,page,content,metadata,
                1 - (embedding <=> %s::vector) AS score
                FROM {self.table} WHERE strategy = %s AND metadata @> %s
                ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s''',
                (vector, strategy, Jsonb(filters or {}), vector, top_k)).fetchall()
        return [dict(zip(('chunk_id', 'document_id', 'source', 'page', 'text', 'metadata', 'score'), row)) for row in rows]
