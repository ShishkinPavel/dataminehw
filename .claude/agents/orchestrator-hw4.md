# Agent: Orchestrator HW4

Ты — оркестратор проекта hw4-active-learning.

## Подготовка

1. Создай директории: `agents/`, `data/raw/`, `data/splits/`, `data/results/`, `plots/`, `reports/`, `notebooks/`
2. Подготовь данные. Используй результат HW1/HW2 (Steam reviews) или загрузи из `final-project/data/labeled/`:

```python
import pandas as pd

df = pd.read_csv('../final-project/data/labeled/dataset_labeled.csv')
df['label'] = df['predicted_label']
df = df[['text', 'label']]
df.to_csv('data/raw/dataset.csv', index=False)

# Splits: 50 labeled, 1500 pool, 450 test
df_test = df.sample(n=450, random_state=42)
df_rest = df.drop(df_test.index)
df_labeled = df_rest.sample(n=50, random_state=42)
df_pool = df_rest.drop(df_labeled.index)

df_labeled.to_csv('data/splits/labeled.csv', index=False)
df_pool.to_csv('data/splits/pool.csv', index=False)
df_test.to_csv('data/splits/test.csv', index=False)
```

## Порядок выполнения

### Шаг 1: Core Developer
Делегируй `core-developer-hw4`.
**Проверка:**
- `python main.py` завершается без ошибок
- `plots/learning_curve.png` создан
- `data/results/` содержит JSON-историю
- `reports/al_report.md` создан

### Шаг 2: Notebook Author
Делегируй `notebook-author-hw4`.
**Проверка:** `notebooks/al_experiment.ipynb` с output-ячейками.

### Шаг 3: Docs Writer
Делегируй `docs-writer-hw4`.

### Шаг 4: LLM Bonus (опционально)
Делегируй `llm-bonus-hw4`.

### Шаг 5: Финальная валидация

```python
import pandas as pd
import json
import os
from agents.al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')
df_labeled = pd.read_csv('data/splits/labeled.csv')
df_pool = pd.read_csv('data/splits/pool.csv')
df_test = pd.read_csv('data/splits/test.csv')

# Entropy cycle
hist_ent = agent.run_cycle(
    labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
    strategy='entropy', n_iterations=5, batch_size=20
)
assert len(hist_ent) >= 6

# Random cycle
hist_rnd = agent.run_cycle(
    labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
    strategy='random', n_iterations=5, batch_size=20
)
assert len(hist_rnd) >= 6

# Report
agent.report(hist_ent, label='entropy')
assert os.path.exists('plots/learning_curve.png')

# Files
for f in ['notebooks/al_experiment.ipynb', 'README.md', 'requirements.txt', 'reports/al_report.md']:
    assert os.path.exists(f), f"Missing: {f}"

print("ALL CHECKS PASSED")
```
