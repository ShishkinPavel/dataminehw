# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sentiment analysis pipeline for movie reviews (IMDB + Rotten Tomatoes). Four agents from homework assignments (HW1–HW4) are unified into a single pipeline in `final-project/`. Each `hw*/` directory contains the original standalone agent; `final-project/agents/` contains synced copies used by the integrated pipeline.

## Commands

```bash
# Install dependencies (Python 3.10+)
pip install -r final-project/requirements.txt

# Run full pipeline (interactive, requires user input for HITL)
cd final-project && python run_pipeline.py
cd final-project && python run_pipeline.py --imdb-size 5000 --rt-size 1000

# Streamlit dashboard (requires pipeline artifacts to exist)
cd final-project && streamlit run dashboard.py
```

There are no tests. Validation is done through pipeline execution.

## Architecture

### Pipeline flow (run_pipeline.py)
```
DataCollectionAgent → DataQualityAgent → AnnotationAgent → ActiveLearningAgent → Train → Reports
     (HW1)              (HW2)              (HW3)              (HW4)
```

### Agents

All agents follow the same pattern: `__init__(config)` loads YAML, public methods are skills.

- **DataCollectionAgent** (`agents/data_collection_agent.py`): `run(sources)` orchestrates `_load_dataset()` (HF library) and `_fetch_hf_api()` (HF REST API). Output schema: `text, label, source, collected_at`.
- **DataQualityAgent** (`agents/data_quality_agent.py`): `detect_issues(df)` → `fix(df, strategy)` → `compare(before, after)`. Strategies are dicts like `{'missing': 'drop', 'duplicates': 'drop', 'outliers': 'clip_iqr'}`.
- **AnnotationAgent** (`agents/annotation_agent.py`): `auto_label(df)` runs BART zero-shot (`facebook/bart-large-mnli`), adds `predicted_label` and `confidence` columns. Never overwrites `label` (GT). `check_quality()` computes Cohen's kappa.
- **ActiveLearningAgent** (`agents/al_agent.py`): `run_cycle(labeled, pool, test, strategy, n_iterations, batch_size)` returns history list. Strategies: `entropy`, `least_confidence`, `margin`, `random`. Note: entropy ≡ least_confidence ≡ margin for binary classification.

### Label terminology (critical)
- **GT (label)** — ground truth from IMDB/HF source
- **BART predicted (predicted_label)** — zero-shot classifier output
- **Final label** — what the model trains on (user decides at annotation step)

Never confuse these. Always specify type when mentioning label distributions.

### Using agents from scripts

Agents live in `hw*/agents/` (source of truth). To use them in scripts outside their directory, add the hw dir to `sys.path` and create a temp YAML config:

```python
import sys, yaml, tempfile
sys.path.insert(0, 'hw1-data-collection')
from agents.data_collection_agent import DataCollectionAgent

config = {'sources': [...], 'output': {'path': 'final-project/data/raw/dataset.csv'}}
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
yaml.dump(config, tmp); tmp.close()
agent = DataCollectionAgent(config=tmp.name)
```

### Claude Skills (`.claude/skills/`)
The `/pipeline` skill orchestrates `/collect` → `/clean` → `/annotate` → `/active-learn` sequentially. Each skill maps to one agent. HITL points use `AskUserQuestion` tool, never plain text questions.

## Key Conventions

- Data artifacts go to `final-project/data/`, `plots/`, `models/`, `reports/`
- `.gitignore` excludes `*.csv`, `*.png`, `*.joblib` — these are generated artifacts, never committed
- Each `hw*/` dir has its own `config.yaml` for standalone runs; `final-project/run_pipeline.py` builds configs inline and writes temp YAML files that are deleted after use
- LLM features (YandexGPT) require `YANDEX_API_KEY` and `YANDEX_FOLDER_ID` in `final-project/.env`. Load with `dotenv.load_dotenv('final-project/.env')` before calling LLM methods
- After each pipeline step, output a status line: `Строк: N | GT: X pos / Y neg | BART: A pos / B neg | Final: ...`
- When syncing agents, copy from `hw*/agents/` → `final-project/agents/` (hw dirs are source of truth)
- The Streamlit dashboard (7 tabs) reads generated artifacts; it requires the pipeline to have been run at least once
- Reports and data cards must be written in Russian (technical terms like F1, Cohen's kappa stay in English)
- IMDB labels arrive as integers (0/1) — convert to strings ('negative'/'positive') immediately after collection
