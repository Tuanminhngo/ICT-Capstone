# Agent 2 — Embedding & Scoring

Production-oriented Python pipeline that encodes paired video clips and thought-report text into a shared semantic space and computes cosine similarity scores for every pair.

## What it does

- Reads `paired_dataset.csv` produced by Agent 1.
- Pre-flight validates every row (skips Agent-1-invalid rows and missing clips).
- **Model comparison** : scores a configurable sample of rows with both CLIP and ImageBind, exporting `model_comparison.csv` to satisfy the "evaluate ≥ 2 models" requirement.
- **Full-dataset scoring**: encodes every valid (clip, thought-text) pair with the primary model (CLIP) and writes cosine similarity scores.
- Caches video embeddings to disk so re-runs skip already-processed clips.
- Exports `similarity_scores.csv`, `similarity_scores.json`, `model_comparison.csv`, `validation_report.csv`, and `summary_report.json`.

## Expected project layout

```text
Vivian Project/
├── Video 1/
├── Video 2/
├── thought_reports.xlsx
├── agent1_data_preparation/
│   └── output/
│       ├── paired_dataset.csv        ← required input for Agent 2
│       └── clips/
│           ├── 5s/
│           ├── 10s/
│           └── 15s/
└── agent2_embedding_scoring/
    ├── main.py
    ├── config.py
    ├── data_loader.py
    ├── encoders.py
    ├── scorer.py
    ├── validator.py
    ├── exporter.py
    ├── requirements.txt
    ├── README.md
    └── output/
        ├── embeddings_cache/
        │   └── clip/
        ├── similarity_scores.csv
        ├── similarity_scores.json
        ├── model_comparison.csv
        ├── validation_report.csv
        └── summary_report.json
```

## Install

```bash
cd agent2_embedding_scoring
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install CLIP (not on PyPI):
pip install git+https://github.com/openai/CLIP.git

# Install ImageBind (comparison model only — optional):
pip install git+https://github.com/facebookresearch/ImageBind.git
```

## Run

```bash
cd agent2_embedding_scoring
python3 main.py
```

## Output schema

`similarity_scores.csv` / `.json` — **one row per (thought_report × window_size × model)**:

| Column | Description |
|---|---|
| `row_id` | Matches Agent 1's `row_id` |
| `thought_text` | Original thought report text |
| `window_size_seconds` | 5, 10, or 15 |
| `model_name` | `"clip"` (or `"imagebind"` in model_comparison.csv) |
| `similarity_score` | Cosine similarity in [−1, 1]; NaN on failure |
| `video_embedding_source` | `"cache"` or `"computed"` |
| `scoring_status` | `"scored"`, `"skipped"`, or `"failed"` |
| `scoring_message` | Empty on success; reason string on failure |
| `clip_path` | Path to the source clip file |

`model_comparison.csv` — same schema, contains both `clip` and `imagebind` rows for the sample subset.

## Key settings (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `primary_model` | `"clip"` | Model used for the full dataset run |
| `clip_model_variant` | `"ViT-B/32"` | CLIP variant; `"ViT-L/14"` is more accurate but slower |
| `comparison_models` | `["imagebind"]` | Models scored on the sample subset |
| `model_comparison_sample_size` | `20` | Unique row IDs in the comparison sample; set to `0` to skip |
| `frames_per_clip` | `8` | Frames sampled per clip; averaged into one embedding |
| `frame_batch_size` | `32` | Reduce if you hit memory pressure on CPU |
| `cache_embeddings` | `True` | Persist video embeddings to disk |
| `overwrite_cache` | `False` | Force re-encoding even if cache exists |
| `skip_invalid_rows` | `True` | Skip rows Agent 1 marked invalid |
| `skip_missing_clips` | `True` | Skip rows where the clip file does not exist |
| `dry_run` | `False` | Validate and log without encoding or writing scores |

## Model selection rationale

Two models were evaluated for this task:

**CLIP (ViT-B/32)** — *selected as primary model*
- Encodes image frames and text into a shared 512-dimensional space.
- Video handled via frame-averaging: N evenly-spaced frames are encoded and their embeddings are averaged. For 5–15 s educational clips, temporal ordering matters less than visual concept coverage, making frame-averaging well-suited to this task.
- ~350 MB model size; 2–6 s per clip on CPU; ~1–2 GB RAM.
- Well-validated in mind-wandering + video research literature.
- Simple installation via HuggingFace / OpenAI GitHub.

**ImageBind (huge)** — *evaluated on sample only*
- Native temporal video encoder; 1024-dimensional shared space across 6 modalities.
- ~5 GB model size; 30–120 s per clip on CPU; ~8–12 GB RAM.
- Impractical for a full dataset run on CPU-only hardware.
- Included in `model_comparison.csv` to satisfy the brief's "evaluate ≥ 2 models" requirement.


