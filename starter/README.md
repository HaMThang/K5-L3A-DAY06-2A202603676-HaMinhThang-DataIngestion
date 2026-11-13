# PostgreSQL + pgvector pipeline

Start database from starter/: `docker compose -p day6_student_starter up -d --wait`. Connection settings are loaded from starter/.env. Database initialization enables the vector extension on a new volume. Keep existing database volumes when troubleshooting.

# Implemented pipeline

No new functions or methods were added. Existing functions now implement reading/normalization, stable chunks, local multilingual embeddings, pgvector snapshots, exact cosine retrieval and conservative evidence evaluation.

Use Python 3.12 (the existing Python 3.15 installation may fail loading DLLs and embedding dependencies). Install requirements into your activated virtual environment. The embedding adapter supports local Sentence Transformers and OpenAI. Local default: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Local inference needs no API key; optionally pin EMBEDDING_REVISION. First local ingestion downloads the real model. Inputs exceeding the token limit fail explicitly rather than being silently truncated.

Run from starter/:

```powershell
python -m pip install -r requirements.txt
python -m src.run --stage prepare
python -m src.run --stage ingest --run-id baseline
python -m src.run --stage dev --run-id baseline
```

Review dev traces and fill manual labels. Automatic selection is available through `python -m src.run --stage select --run-id baseline`: the rule is frozen before evaluation (dev Evidence Hit@3, then Hit@1, then median query embedding + search latency). Alternatively create selection.json with selected_strategy and an evidence-based reason. Then run:

```powershell
python -m src.run --stage test --run-id baseline
```

Default command without arguments only prepares data. Prepare does not load the model or access the database. Arguments: --fixed-size 500 --overlap 50 --paragraph-size 1000; use the same arguments at ingestion and evaluation. Change run-id for new experiments. A frozen experiment rejects corpus, benchmark, code, model or chunk-setting changes. Dependency versions and resolved model revision are checked at evaluation.

Canonical input is Arduino official Markdown and Python main HTML content. Original Arduino website HTML and derived HTML are not ingested again. Benchmark page.text is never used as an ingestion source. Heading IDs match gold_page. The manifest records SHA-256, section IDs and character counts; only sections with >=1500 normalized characters count toward the minimum (smaller meaningful sections remain searchable). Raw sources are preserved.

Each run writes corpus_manifest.json, pages.json, preparation.json, errors.log, frozen_config.json after successful ingestion, and dev/test traces and summaries after evaluation. Ingest validates persistence and repeats snapshot replacement to check no duplicates. upsert accepts a COMPLETE strategy snapshot, removes obsolete chunks within that namespace transactionally, and never deletes another run's strategy. Tables are separated by dimension (`ingestion_chunks_384`, `_1536`, `_3072`); each run/strategy has a separate namespace. Existing tables are never dropped automatically.

Results include actual tokenizer/API token counts, estimated API costs, character distributions, separate embedding/ingestion times, exact cosine search and full source metadata. B and C share document embeddings; A/B/C share each query embedding. Local hardware/electricity is excluded. Coherence and human grounding/metadata labels are intentionally blank; do not report them as completed manual evaluation. Exact-match evidence scores are conservative and may miss semantically equivalent evidence: review and document alternatives.

## Comparing embedding models and optional LLM answers

Keep `.env` private; the OpenAI adapter reads `OPENAI_API_KEY` without printing or persisting it. Select another provider/model using CLI options rather than changing `.env` between experiments:

```powershell
python -m src.run --stage ingest --run-id openai_small --provider openai --model text-embedding-3-small --generate-answers
python -m src.run --stage dev --run-id openai_small --provider openai --model text-embedding-3-small --generate-answers
python -m src.run --stage select --run-id openai_small
python -m src.run --stage test --run-id openai_small --provider openai --model text-embedding-3-small --generate-answers
```

Repeat with a new run-id and `text-embedding-3-large`, then compare:

```powershell
python -m src.run --stage compare --run-id openai_comparison --compare-runs openai_small openai_large
```

Or run `./scripts/compare_openai.ps1 -Python python -RunPrefix experiment2` to execute both full experiments in sequence. Use a fresh prefix after code/data/model changes. For local comparison use `--provider local --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` in a separate run; local model/dependencies must first be downloaded. `--dimensions` can shorten supported OpenAI embeddings, but use a new run-id and report the dimension as part of the experiment.

`--generate-answers` uses the same LLM (`gpt-5.6-terra` by default; override `--llm-model`) and prompt for every strategy/model. LLM sees ONLY the question and retrieved chunks, never gold answer/evidence. Structured output contains Vietnamese answer, insufficient-evidence flag and exact-quote citations. Citation checks do not prove entailment or answer accuracy. Generated answers, latency, model, token usage and estimated costs are stored in traces. Test results are generated only after a dev selection file exists. No LLM judge or fabricated manual scores are used.

Costs use official rates checked 2026-11-13: embedding small $0.02/M input tokens, large $0.13/M; GPT-5.6 Terra input $2/M, cached input $0.2/M, output $12/M. This is an estimate based on successful responses, not a billing export; probes/retries and failed calls are excluded. Pricing for another LLM remains unknown rather than guessed. Sources: [embedding small](https://developers.openai.com/api/docs/models/text-embedding-3-small), [embedding large](https://developers.openai.com/api/docs/models/text-embedding-3-large), [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra).

Run `python scripts/verify_pipeline.py --database` for invariants and an isolated synthetic pgvector integration test. Synthetic vectors are used only in this explicit unit test, never in real ingestion/evaluation.
