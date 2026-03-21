# Agent: Docs Writer HW4

Создай `README.md` для hw4-active-learning.

## Структура

```markdown
# ActiveLearningAgent — HW4

Агент для умного отбора данных (Active Learning). Сравнивает стратегии отбора и показывает экономию разметки.
Домашнее задание №4, Трек A.

## Задача ML

Sentiment classification на текстовых отзывах. Active Learning позволяет достичь того же качества модели с меньшим количеством размеченных примеров.

## Архитектура агента

```
ActiveLearningAgent(model='logreg')
│
├── fit(labeled_df)              → обучить TF-IDF + classifier
├── query(pool, strategy)        → выбрать batch_size примеров
│   ├── entropy                  наиб. энтропия предсказаний
│   ├── margin                   мин. разница между top-2 классами
│   └── random                   случайная выборка (baseline)
├── evaluate(test_df)            → accuracy, F1
├── run_cycle(...)               → полный AL-цикл с историей
├── report(history)              → learning_curve.png
└── compare_strategies(...)      → strategy_comparison.png
```

## AL-цикл

1. Старт: 50 размеченных примеров
2. 5 итераций по 20 примеров
3. На каждой итерации: fit → query → add to labeled → evaluate
4. Финал: 150 размеченных примеров

## Результаты

### Сравнение стратегий
(вставить таблицу из reports/al_report.md)

### Learning Curves
![Strategy Comparison](plots/strategy_comparison.png)

### Экономия
(вставить данные об экономии)

## Быстрый старт

```bash
cd hw4-active-learning
pip install -r requirements.txt
python main.py
```

## Использование

```python
from agents.al_agent import ActiveLearningAgent

agent = ActiveLearningAgent(model='logreg')
history = agent.run_cycle(
    labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
    strategy='entropy', n_iterations=5, batch_size=20
)
agent.report(history)
```

## Структура репозитория
(дерево)

## Требования
Python 3.10+, зависимости: requirements.txt
```

## Правила
- Прочитай reports/al_report.md и вставь реальные цифры
- Если plots/strategy_comparison.png существует — сослаться на него
