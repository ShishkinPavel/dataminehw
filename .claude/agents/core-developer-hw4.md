# Agent: Core Developer HW4

Ты — разработчик ядра ActiveLearningAgent.

## Файлы

### 1. agents/al_agent.py

Класс `ActiveLearningAgent`:

**Конструктор:**
- `__init__(self, model: str = 'logreg', config: str | None = None)`
- model: 'logreg' | 'svm' | 'rf' (LogisticRegression / SVC / RandomForestClassifier)
- Внутренне создаёт sklearn Pipeline: TfidfVectorizer + classifier
- self.pipeline = None (создаётся в fit)
- self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

---

#### Skill 1: fit(labeled_df: pd.DataFrame) -> None

Обучает модель на размеченных данных.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

self.pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
self.pipeline.fit(labeled_df['text'], labeled_df['label'])
```

- Выбор модели по self.model_name
- Логирует: кол-во примеров, кол-во классов

---

#### Skill 2: query(pool_df: pd.DataFrame, strategy: str = 'entropy', batch_size: int = 20) -> list[int]

Выбирает наиболее информативные примеры из пула.

Стратегии:

**'entropy'** — максимальная энтропия предсказаний:
```python
proba = self.pipeline.predict_proba(pool_df['text'])
entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
indices = np.argsort(entropy)[-batch_size:]  # top-N by entropy
```

**'margin'** — минимальный margin между двумя верхними классами:
```python
proba = self.pipeline.predict_proba(pool_df['text'])
sorted_proba = np.sort(proba, axis=1)
margin = sorted_proba[:, -1] - sorted_proba[:, -2]
indices = np.argsort(margin)[:batch_size]  # smallest margin
```

**'random'** — случайная выборка (baseline):
```python
indices = np.random.choice(len(pool_df), size=batch_size, replace=False)
```

Возвращает: список индексов (позиции в pool_df).

---

#### Skill 3: evaluate(test_df: pd.DataFrame) -> dict

```python
from sklearn.metrics import accuracy_score, f1_score

preds = self.pipeline.predict(test_df['text'])
return {
    'accuracy': accuracy_score(test_df['label'], preds),
    'f1': f1_score(test_df['label'], preds, average='weighted'),
    'predictions': preds  # для анализа
}
```

---

#### Skill 4: run_cycle(labeled_df, pool_df, test_df, strategy, n_iterations, batch_size) -> list[dict]

Полный AL-цикл:

```python
def run_cycle(
    self,
    labeled_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    test_df: pd.DataFrame,
    strategy: str = 'entropy',
    n_iterations: int = 5,
    batch_size: int = 20
) -> list[dict]:
    history = []
    current_labeled = labeled_df.copy()
    current_pool = pool_df.copy()

    for i in range(n_iterations + 1):  # +1 для initial evaluation
        # Fit
        self.fit(current_labeled)

        # Evaluate
        metrics = self.evaluate(test_df)

        # Record
        record = {
            'iteration': i,
            'n_labeled': len(current_labeled),
            'accuracy': round(metrics['accuracy'], 4),
            'f1': round(metrics['f1'], 4),
            'strategy': strategy
        }
        history.append(record)
        logger.info(f"Iter {i}: n={len(current_labeled)}, acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")

        # Query (skip on last iteration)
        if i < n_iterations and len(current_pool) >= batch_size:
            indices = self.query(current_pool, strategy=strategy, batch_size=batch_size)
            selected = current_pool.iloc[indices]
            current_labeled = pd.concat([current_labeled, selected], ignore_index=True)
            current_pool = current_pool.drop(current_pool.index[indices]).reset_index(drop=True)

    return history
```

---

#### Skill 5: report(history, label='', output_dir='plots') -> None

Генерирует learning curve.

```python
import matplotlib.pyplot as plt

def report(self, history: list[dict], label: str = '', output_dir: str = 'plots') -> None:
    os.makedirs(output_dir, exist_ok=True)
    df_hist = pd.DataFrame(history)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curve
    axes[0].plot(df_hist['n_labeled'], df_hist['accuracy'], 'o-', label=label or df_hist['strategy'].iloc[0])
    axes[0].set_xlabel('Number of labeled examples')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Learning Curve — Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # F1 curve
    axes[1].plot(df_hist['n_labeled'], df_hist['f1'], 's-', label=label or df_hist['strategy'].iloc[0])
    axes[1].set_xlabel('Number of labeled examples')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Learning Curve — F1')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curve.png', dpi=150)
    plt.close()
```

Также добавь метод для сравнения нескольких стратегий на одном графике:

```python
@staticmethod
def compare_strategies(histories: dict[str, list[dict]], output_dir: str = 'plots') -> None:
    """Plot multiple strategies on one chart.

    Args:
        histories: {'entropy': history_list, 'random': history_list, ...}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'entropy': 'tab:blue', 'margin': 'tab:orange', 'random': 'tab:gray'}

    for name, history in histories.items():
        df_h = pd.DataFrame(history)
        c = colors.get(name, None)
        axes[0].plot(df_h['n_labeled'], df_h['accuracy'], 'o-', label=name, color=c)
        axes[1].plot(df_h['n_labeled'], df_h['f1'], 's-', label=name, color=c)

    for ax, metric in zip(axes, ['Accuracy', 'F1']):
        ax.set_xlabel('Number of labeled examples')
        ax.set_ylabel(metric)
        ax.set_title(f'Active Learning — {metric}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=150)
    plt.close()
```

---

**Требования к коду:**
- Type hints, Google-style docstrings
- logging
- try/except
- Копии DataFrame (не мутировать оригинал)
- Фиксированные random seeds для воспроизводимости

### 2. config.yaml

```yaml
model: logreg
tfidf:
  max_features: 5000
  ngram_range: [1, 2]

active_learning:
  initial_size: 50
  batch_size: 20
  n_iterations: 5
  strategies:
    - entropy
    - margin
    - random

paths:
  raw: data/raw/dataset.csv
  labeled: data/splits/labeled.csv
  pool: data/splits/pool.csv
  test: data/splits/test.csv
  results: data/results/
  plots: plots/
  report: reports/al_report.md
```

### 3. main.py

```python
"""Active Learning experiment. Запуск: python main.py"""
import logging
import json
import pandas as pd
from agents.al_agent import ActiveLearningAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

def main():
    df_labeled = pd.read_csv('data/splits/labeled.csv')
    df_pool = pd.read_csv('data/splits/pool.csv')
    df_test = pd.read_csv('data/splits/test.csv')

    agent = ActiveLearningAgent(model='logreg', config='config.yaml')
    histories = {}

    # Run entropy strategy
    print("=" * 60)
    print("STRATEGY: entropy")
    print("=" * 60)
    hist_entropy = agent.run_cycle(
        labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
        strategy='entropy', n_iterations=5, batch_size=20
    )
    histories['entropy'] = hist_entropy

    # Run margin strategy
    print("\n" + "=" * 60)
    print("STRATEGY: margin")
    print("=" * 60)
    hist_margin = agent.run_cycle(
        labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
        strategy='margin', n_iterations=5, batch_size=20
    )
    histories['margin'] = hist_margin

    # Run random baseline
    print("\n" + "=" * 60)
    print("STRATEGY: random")
    print("=" * 60)
    hist_random = agent.run_cycle(
        labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
        strategy='random', n_iterations=5, batch_size=20
    )
    histories['random'] = hist_random

    # Compare
    ActiveLearningAgent.compare_strategies(histories)
    print("\nStrategy comparison saved to plots/strategy_comparison.png")

    # Save results
    import os
    os.makedirs('data/results', exist_ok=True)
    with open('data/results/histories.json', 'w') as f:
        json.dump(histories, f, indent=2)

    # Report
    os.makedirs('reports', exist_ok=True)
    final_ent = hist_entropy[-1]
    final_rnd = hist_random[-1]

    # Найти сколько примеров экономит entropy vs random для того же качества
    target_acc = final_rnd['accuracy']
    saved = 0
    for rec in hist_entropy:
        if rec['accuracy'] >= target_acc:
            saved = final_rnd['n_labeled'] - rec['n_labeled']
            break

    with open('reports/al_report.md', 'w') as f:
        f.write('# Active Learning Report\n\n')
        f.write('## Strategies Compared\n\n')
        f.write('| Strategy | Final Accuracy | Final F1 | N Labeled |\n')
        f.write('|----------|---------------|----------|----------|\n')
        for name, hist in histories.items():
            final = hist[-1]
            f.write(f"| {name} | {final['accuracy']:.4f} | {final['f1']:.4f} | {final['n_labeled']} |\n")
        f.write(f'\n## Savings\n\n')
        f.write(f'Entropy strategy reached random baseline accuracy ({target_acc:.4f}) ')
        f.write(f'with **{saved} fewer labeled examples**.\n')

    print(f"\nReport saved to reports/al_report.md")
    print(f"Entropy saved ~{saved} examples vs random baseline")

if __name__ == '__main__':
    main()
```

### 4. requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0
jupyter>=1.0.0
datasets>=2.14.0
tabulate>=0.9.0
anthropic>=0.25.0
```

### 5. .gitkeep в: data/raw/, data/splits/, data/results/, plots/, reports/

## Валидация

1. `pip install -r requirements.txt`
2. Подготовь данные (оркестратор сделает)
3. `python main.py` — 3 стратегии, графики, отчёт
4. `plots/strategy_comparison.png` — 3 кривые на одном графике
