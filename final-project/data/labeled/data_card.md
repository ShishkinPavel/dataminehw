# Data Card — Steam Gaming Reviews (RPG)

## Dataset Overview
- **Task**: Sentiment Analysis (binary: positive/negative)
- **Domain**: Steam gaming reviews for indie RPG games
- **Total examples**: 1311 (after cleaning)
- **Label distribution**: 908 positive / 403 negative
- **Language**: English

## Data Sources

### 1. HuggingFace — `ksang/steamreviews`
- **Type**: Pre-existing dataset
- **Sample size**: 500 reviews
- **Labels**: Derived from `review_score` (1=positive, else=negative)

### 2. Steam Reviews API
- **Type**: Live API scraping
- **Games** (10 RPG indie titles):
  - Castle Crashers, Robocraft, Kathy Rain, Vampire Survivors
  - Clicker Heroes, Hades, Undertale, Slay the Spire, Dead Cells
- **Reviews per game**: up to 100
- **Labels**: Based on `voted_up` flag (thumbs up/down)

## Data Processing

### Cleaning (HW2 — DataQualityAgent)
- **Missing values**: 3 rows dropped (empty text)
- **Duplicates**: 86 rows removed (6.14%)
- **Outliers**: None detected (text-only data)
- **Class imbalance**: ratio=0.130 (strong imbalance → addressed with balanced class weights)

### Labeling (HW3 — AnnotationAgent)
- **Method**: Zero-shot classification via `facebook/bart-large-mnli`
- **Mean confidence**: 0.899
- **Low confidence (<0.7)**: 172 examples (13.1%)
- **Agreement with ground truth**: 77.6%
- **Cohen's κ**: 0.370

### Human Review (HITL)
- **Reviewed**: 10 low-confidence examples
- **Corrected**: 6 labels (mainly false negatives → positive)
- **Patterns**: Short/ambiguous texts, sarcasm, mixed sentiment

## Schema
| Column | Type | Description |
|--------|------|-------------|
| text | string | Review text |
| label | string | Ground truth label (positive/negative) |
| source | string | Data source identifier |
| collected_at | datetime | Collection timestamp |
| predicted_label | string | Zero-shot model prediction |
| confidence | float | Prediction confidence [0, 1] |

## Limitations
- Strong class imbalance (69% positive vs 31% negative)
- English-only reviews
- Limited to RPG/indie genre — may not generalize
- Zero-shot labeling accuracy ~78% — some label noise remains
- Short reviews (<10 words) are harder to classify reliably

## Ethical Considerations
- Reviews are publicly available on Steam
- No personally identifiable information (PII) retained
- Steam-censored profanity appears as ♥♥♥♥
