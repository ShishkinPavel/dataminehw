# Agent: Notebook Author HW4

Создай `notebooks/al_experiment.ipynb` через nbformat.

## Секции ноутбука

### 1. Setup и загрузка данных
```python
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from agents.al_agent import ActiveLearningAgent

plt.style.use('seaborn-v0_8-whitegrid')

df_labeled = pd.read_csv('../data/splits/labeled.csv')
df_pool = pd.read_csv('../data/splits/pool.csv')
df_test = pd.read_csv('../data/splits/test.csv')

print(f"Labeled: {len(df_labeled)}, Pool: {len(df_pool)}, Test: {len(df_test)}")
```

### 2. Обзор начальной выборки
- Распределение классов в labeled, pool, test (bar charts)
- Распределение длин текстов
- "Начинаем с всего {N} размеченных примеров"

### 3. Baseline: модель на начальных данных
```python
agent = ActiveLearningAgent(model='logreg')
agent.fit(df_labeled)
baseline = agent.evaluate(df_test)
print(f"Baseline (N={len(df_labeled)}): accuracy={baseline['accuracy']:.4f}, f1={baseline['f1']:.4f}")
```

### 4. AL-цикл: Entropy
```python
hist_entropy = agent.run_cycle(
    labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
    strategy='entropy', n_iterations=5, batch_size=20
)
pd.DataFrame(hist_entropy)
```
- Таблица с итерациями

### 5. AL-цикл: Margin
```python
hist_margin = agent.run_cycle(
    labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
    strategy='margin', n_iterations=5, batch_size=20
)
```

### 6. AL-цикл: Random (baseline)
```python
hist_random = agent.run_cycle(
    labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
    strategy='random', n_iterations=5, batch_size=20
)
```

### 7. Сравнение стратегий — Learning Curves
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, hist, color, marker in [
    ('entropy', hist_entropy, 'tab:blue', 'o'),
    ('margin', hist_margin, 'tab:orange', 's'),
    ('random', hist_random, 'tab:gray', '^')
]:
    df_h = pd.DataFrame(hist)
    axes[0].plot(df_h['n_labeled'], df_h['accuracy'], f'{marker}-', label=name, color=color)
    axes[1].plot(df_h['n_labeled'], df_h['f1'], f'{marker}-', label=name, color=color)

for ax, title in zip(axes, ['Accuracy', 'F1 Score']):
    ax.set_xlabel('Number of labeled examples')
    ax.set_ylabel(title)
    ax.set_title(f'Active Learning — {title}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('../plots/strategy_comparison.png', dpi=150)
plt.show()
```

### 8. Анализ: сколько примеров экономит AL
Markdown + код:
- Для каждого уровня accuracy (baseline random): сколько примеров нужно entropy vs random
- Таблица: target_accuracy | n_entropy | n_random | saved
- Bar chart savings
- "Entropy достигает accuracy X с Y меньшим количеством примеров"

### 9. Анализ отобранных примеров
- Какие тексты entropy выбирает первыми?
- Длина текстов, распределение label в отобранных
- "Entropy предпочитает неоднозначные примеры"

### 10. Сводная таблица результатов
| Strategy | Start Acc | Final Acc | Start F1 | Final F1 | N Labeled |
- pd.DataFrame из histories

### 11. Выводы
Markdown:
- Entropy vs Random: на сколько эффективнее
- Margin vs Entropy: в чём разница
- Практическая рекомендация: когда использовать AL
- Ограничения эксперимента

## Стиль
- figsize >= (12, 5), заголовки, подписи, tight_layout
- Set2 или tab:blue/orange/gray для стратегий
- Таблицы через pd.DataFrame.style или print

## Валидация
```bash
cd notebooks && jupyter nbconvert --to notebook --execute al_experiment.ipynb --output al_experiment.ipynb
```
