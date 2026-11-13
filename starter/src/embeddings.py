"""Local or OpenAI embeddings through the existing adapter contract."""
import math
import os
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class MyEmbedder:
    def __init__(self, model_name: str, provider: str = 'local', dimensions: int | None = None) -> None:
        self.model_name = model_name
        self.provider = provider
        self.prompt_tokens = 0
        self.api_requests = 0
        self.resolved_revision = None
        self.batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', '32'))
        if not 1 <= self.batch_size <= 128:
            raise ValueError('EMBEDDING_BATCH_SIZE must be between 1 and 128')
        if provider == 'openai':
            from openai import OpenAI
            import tiktoken
            limits = {'text-embedding-3-small': 1536, 'text-embedding-3-large': 3072}
            if model_name not in limits:
                raise ValueError('Supported OpenAI embeddings: text-embedding-3-small / text-embedding-3-large')
            self.dimensions = limits[model_name] if dimensions is None else dimensions
            if not 1 <= self.dimensions <= limits[model_name]:
                raise ValueError('Invalid embedding dimensions')
            self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.openai.com/v1',
                                 timeout=60, max_retries=1)
            self.tokenizer = tiktoken.get_encoding('cl100k_base')
        elif provider == 'local':
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, revision=os.getenv('EMBEDDING_REVISION') or None)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            if not self.dimensions or dimensions not in (None, self.dimensions):
                raise ValueError('Requested dimensions do not match local model')
            self.model.max_seq_length = min(512, self.model[0].auto_model.config.max_position_embeddings)
            self.resolved_revision = getattr(self.model[0].auto_model.config, '_commit_hash', None)
        else:
            raise ValueError('Embedding provider must be local or openai')

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError('Empty embedding input')
        if self.provider == 'openai':
            lengths = [len(self.tokenizer.encode(text, disallowed_special=())) for text in texts]
            if any(length > 8191 for length in lengths):
                raise ValueError('OpenAI embedding input exceeds conservative 8191-token limit')
            vectors = []
            for start in range(0, len(texts), self.batch_size):
                response = self.client.embeddings.create(model=self.model_name,
                    input=texts[start:start + self.batch_size], dimensions=self.dimensions, encoding_format='float')
                batch = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in batch] != list(range(len(texts[start:start + self.batch_size]))):
                    raise RuntimeError('Embedding response count/order mismatch')
                for item in batch:
                    if len(item.embedding) != self.dimensions or not all(math.isfinite(v) for v in item.embedding):
                        raise RuntimeError('Invalid API embedding')
                    norm = math.sqrt(sum(v * v for v in item.embedding))
                    if not norm:
                        raise RuntimeError('Zero embedding returned')
                    vectors.append([v / norm for v in item.embedding])
                self.prompt_tokens += response.usage.prompt_tokens
                self.api_requests += 1
            return vectors
        lengths = self.model.tokenizer(texts, truncation=False, return_length=True)['length']
        if any(length > self.model.max_seq_length for length in lengths):
            raise ValueError('Input exceeds model token limit; reduce chunk size instead of truncating evidence')
        self.prompt_tokens += sum(lengths)
        return self.model.encode(texts, batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
