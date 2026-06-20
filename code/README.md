# Solution — Multi-Stage Claude Pipeline

Verifies damage claims by running images and claim history through a three-stage Claude pipeline, then writes structured predictions to `output.csv`.

---

## Setup

**1. Install dependencies** (Python 3.10+):

```bash
pip install -r requirements.txt
```

**2. Set your Anthropic API key:**

```bash
cp .env.example .env
# then edit .env and paste your key
```

Or export it directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # macOS/Linux
$env:ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell
```

---

## Running the pipeline

**Generate predictions** on `dataset/claims.csv`:

```bash
python code/main.py --verbose
```

Output is written to `dataset/output.csv`. The run is **resume-safe** — if it is interrupted, re-running will skip rows already written.

**Run evaluation** on `dataset/sample_claims.csv` (has ground-truth labels):

```bash
python code/evaluation/main.py --verbose
```

Prints per-field accuracy and writes `code/evaluation/evaluation_report.md`.

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--history-model` | `claude-sonnet-4-6` | Model for Stage 1 (history assessment) |
| `--vision-model` | `claude-opus-4-8` | Model for Stage 2 (image analysis) |
| `--decision-model` | `claude-sonnet-4-6` | Model for Stage 3 (final decision) |
| `--claims` | `dataset/claims.csv` | Path to input CSV |
| `--output` | `dataset/output.csv` | Path to write predictions |
| `--cache` | `code/.cache/responses.json` | Path to disk cache |
| `--verbose`, `-v` | off | Print per-claim progress |

---

## How it works

Each claim is processed in three sequential API calls:

**Stage 1 — History assessment** (`claude-sonnet-4-6`, text-only)
Evaluates user credibility from claim history, past rejections, and risk flags. Returns `history_risk_level` (low/medium/high) and any history-based risk flags.

**Stage 2 — Image analysis** (`claude-opus-4-8`, vision)
Sends all submitted images alongside a structured prompt covering evidence requirements for the claimed object type. Returns damage type, severity, object part, visual claim status, and image-level risk flags.

Supports JPEG, PNG, WebP natively. AVIF and other formats are automatically converted to PNG via Pillow before sending.

**Stage 3 — Final decision** (`claude-sonnet-4-6`, text-only)
Synthesises Stage 1 and Stage 2 outputs into the final verdict. Visual evidence is primary; history risk can trigger `manual_review_required` but cannot override clear image evidence.

All stage outputs are validated and coerced to the allowed value sets before writing to CSV.

### Caching

Every model response is cached to `.cache/responses.json` (keyed by SHA-256 of stage + model + claim inputs). Re-runs are instant for cached claims. Entries with error/fallback values are automatically invalidated so the model is re-called on the next run.

---

## File layout

```
code/
├── main.py                  # Pipeline entry point
├── README.md                # This file
└── evaluation/
    ├── main.py              # Evaluation entry point
    └── evaluation_report.md # Generated after evaluation run
```
