# ActiveLearningAgent — HW4

An intelligent agent for **smart data selection** through Active Learning. This project implements multiple query strategies to identify the most informative examples from an unlabeled pool, reducing the amount of human labeling required while maintaining model performance.

## Description

Active Learning is a machine learning paradigm that selectively labels the most informative examples from a pool of unlabeled data. Instead of random sampling, the agent uses uncertainty and diversity-based strategies to minimize labeling costs while maximizing model performance.

This implementation compares three query strategies:
- **Entropy**: Selects examples with highest prediction uncertainty
- **Margin**: Selects examples where model is least confident in binary distinctions
- **Random**: Baseline random selection for comparison

## ML Task

**Sentiment Classification on Text Reviews**

The agent trains a text classifier to predict sentiment (positive/negative). The pipeline uses:
- TF-IDF vectorization (unigrams + bigrams, 5000 features)
- Logistic Regression classifier (with extensibility to SVM and Random Forest)
- Weighted F1 score and accuracy metrics

## Agent Architecture

```
┌─────────────────────────────────────────────────────┐
│         ActiveLearningAgent                          │
├─────────────────────────────────────────────────────┤
│                                                       │
│  fit()                   Initialize pipeline         │
│  │ - TfidfVectorizer                                 │
│  │ - LogisticRegression (or SVM/RF)                  │
│  └─→ Trained model ready for query                   │
│                                                       │
│  query_strategies()      Select informative examples │
│  │ - entropy: max uncertainty (high entropy)          │
│  │ - margin: smallest confidence margin              │
│  │ - random: baseline                                │
│  └─→ Indices of selected examples                    │
│                                                       │
│  evaluate()              Assess model performance    │
│  │ - accuracy_score                                  │
│  │ - f1_score (weighted)                             │
│  └─→ Metrics dict                                    │
│                                                       │
│  run_cycle()             Full AL loop orchestration  │
│  │ [fit → evaluate → query → add_to_labeled]         │
│  │ Repeat n_iterations times                         │
│  └─→ History of metrics per iteration                │
│                                                       │
│  compare_strategies()    Visualize strategy         │
│  │ Accuracy and F1 curves across strategies          │
│  └─→ plots/strategy_comparison.png                   │
│                                                       │
│  report()                Single strategy learning    │
│  │ Accuracy and F1 vs labeled set size               │
│  └─→ plots/learning_curve.png                        │
│                                                       │
│  llm_explain_selection() Analyze selected examples  │
│  │ Call YandexGPT API to interpret selections        │
│  └─→ Natural language explanation                    │
│                                                       │
│  llm_recommend_strategy() Suggest best approach     │
│  │ Compare all strategies, recommend winner          │
│  └─→ Strategic recommendation                        │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## Active Learning Cycle

The experiment follows this AL protocol:

```
Initial State:   50 labeled examples + ~800 unlabeled pool
                        ↓
Iteration 1:     Fit model on 50 → Select 20 (entropy/margin/random)
                        ↓
Iteration 2:     Fit model on 70 → Select 20
                        ↓
Iteration 3:     Fit model on 90 → Select 20
                        ↓
Iteration 4:     Fit model on 110 → Select 20
                        ↓
Iteration 5:     Fit model on 130 → Select 20
                        ↓
Final State:     150 labeled examples
                 (5 iterations × 20 batch + 50 initial)
```

Total evaluation points: 6 checkpoints (initial + 5 iterations)

## Results

Results are generated at runtime and saved to `reports/al_report.md`. Strategy comparison plots are saved to `plots/strategy_comparison.png`.

Key insight: entropy and margin strategies consistently outperform random baseline, demonstrating that intelligent example selection reduces labeling costs while maintaining model quality.

## Quick Start

### Installation

```bash
# Clone the repository
cd hw4-active-learning

# Install dependencies (Python 3.10+ required)
pip install -r requirements.txt
```

### Run the Experiment

```bash
python main.py
```

This will:
1. Load dataset splits (labeled/pool/test)
2. Run entropy strategy AL cycle (5 iterations)
3. Run margin strategy AL cycle (5 iterations)
4. Run random baseline AL cycle (5 iterations)
5. Generate comparison plots
6. Produce AL report with metrics
7. Display LLM analysis of strategies (if YANDEX_API_KEY set)

Expected output: ~3-5 minutes on CPU

## Usage Example

```python
from agents.al_agent import ActiveLearningAgent
import pandas as pd

# Load data
df_labeled = pd.read_csv('data/splits/labeled.csv')  # 50 examples
df_pool = pd.read_csv('data/splits/pool.csv')         # ~800 examples
df_test = pd.read_csv('data/splits/test.csv')         # test set

# Initialize agent
agent = ActiveLearningAgent(model='logreg', config='config.yaml')

# Run entropy-based AL for 5 iterations, adding 20 examples per iteration
history = agent.run_cycle(
    labeled_df=df_labeled,
    pool_df=df_pool,
    test_df=df_test,
    strategy='entropy',
    n_iterations=5,
    batch_size=20
)

# View learning curve
agent.report(history, label='Entropy Strategy')

# Compare multiple strategies
histories = {
    'entropy': history_entropy,
    'margin': history_margin,
    'random': history_random
}
ActiveLearningAgent.compare_strategies(histories)
```

## Project Structure

```
hw4-active-learning/
├── README.md                          # This file
├── main.py                           # Entry point for experiments
├── config.yaml                       # Configuration (model, AL params)
├── requirements.txt                  # Dependencies
│
├── agents/
│   ├── __init__.py
│   └── al_agent.py                  # ActiveLearningAgent class
│                                      # - fit, query, evaluate
│                                      # - run_cycle, compare_strategies
│                                      # - llm_explain_selection
│                                      # - llm_recommend_strategy
│
├── data/
│   ├── splits/
│   │   ├── labeled.csv               # Initial 50 labeled examples
│   │   ├── pool.csv                  # Unlabeled candidate pool
│   │   └── test.csv                  # Test set
│   └── results/
│       └── histories.json            # Raw metrics from experiment
│
├── plots/
│   ├── strategy_comparison.png       # All strategies side-by-side
│   └── learning_curve.png            # Single strategy curves
│
└── reports/
    └── al_report.md                  # Markdown summary with metrics
```

## Requirements

- **Python 3.10+**
- **pandas** >= 2.0.0 (data manipulation)
- **numpy** >= 1.24.0 (numerical computing)
- **scikit-learn** >= 1.3.0 (ML models, metrics)
- **matplotlib** >= 3.7.0 (plotting)
- **pyyaml** >= 6.0 (config parsing)
- **requests** >= 2.31.0 (HTTP requests for YandexGPT API)

See `requirements.txt` for complete dependency list.

## Bonus: LLM Integration (+1 Point)

This implementation includes intelligent LLM integration using **YandexGPT API**:

### Features

1. **`llm_explain_selection()`** — Analyzes selected examples
   - Explains why entropy/margin strategies choose specific samples
   - Identifies patterns: ambiguous sentiment, sarcasm, mixed opinions
   - Provides human-interpretable reasoning for selections
   - Output in Russian

2. **`llm_recommend_strategy()`** — Strategic recommendation
   - Compares all three strategies based on experiment results
   - Analyzes efficiency: quality per labeled example
   - Quantifies savings: how many labels entropy/margin save vs random
   - Recommends practical strategy for real-world deployment

### Usage

```python
# Explain why certain examples were selected
explanation = agent.llm_explain_selection(
    selected_texts=['Review 1...', 'Review 2...'],
    strategy='entropy',
    iteration=3
)
print(explanation)

# Get strategic recommendation after comparing all strategies
recommendation = agent.llm_recommend_strategy(histories)
print(recommendation)
```

### Requirements for Bonus

Set environment variables before running:

```bash
export YANDEX_API_KEY="your-key-here"
export YANDEX_FOLDER_ID="your-folder-id"
python main.py
```

If API key is not set, the methods gracefully degrade and return informative messages. The main AL experiment runs independently of LLM integration.

---

**Authors**: Active Learning research group
**Version**: 1.0
**License**: Educational use (HW4 submission)
