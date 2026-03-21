# DataQualityAgent — «Детектив данных»

Агент для автоматического обнаружения и устранения проблем качества данных.
Домашнее задание №2 по курсу AI Agents.

## Задача ML

Sentiment analysis отзывов на инди-игры Steam. Агент работает поверх датасета из HW1
и обеспечивает качество данных перед обучением модели.

## Архитектура агента

```
DataQualityAgent
│
├── detect_issues(df)              → QualityReport (dict)
│   ├── missing values analysis
│   ├── duplicates detection
│   ├── outliers (IQR + z-score)
│   └── class imbalance check
│
├── fix(df, strategy)              → pd.DataFrame
│   ├── missing: mean|median|mode|drop|ffill
│   ├── duplicates: drop|keep_first|keep_last
│   └── outliers: clip_iqr|clip_zscore|drop
│
├── compare(before, after)         → pd.DataFrame (метрики)
│
└── llm_recommend(report, task)    → str (бонус: YandexGPT API)
```

## Три части задания

| Часть | Название | Описание |
|-------|----------|----------|
| 1 | Детектив | Обнаружение и визуализация проблем |
| 2 | Хирург | Две стратегии чистки + сравнение |
| 3 | Аргумент | Обоснование выбора лучшей стратегии |

## Обнаруживаемые проблемы

| Проблема | Метод детекции | Стратегии исправления |
|----------|---------------|----------------------|
| Пропуски | isnull() + процент | mean, median, mode, drop, ffill |
| Дубликаты | duplicated() | drop, keep_first, keep_last |
| Выбросы | IQR, z-score | clip_iqr, clip_zscore, drop |
| Дисбаланс | value_counts ratio | (детекция, не исправление) |

## Быстрый старт

```bash
cd hw2-data-quality
pip install -r requirements.txt
python main.py
```

## Использование в коде

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df)
df_clean = agent.fix(df, strategy={'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
comparison = agent.compare(df, df_clean)
```

## Результаты

Генерируются при запуске `python main.py`, сохраняются в `reports/quality_report.md`.

Метод `compare()` выводит таблицу до/после по метрикам: total_rows, missing_values, duplicates, outliers_iqr, label_imbalance_ratio.

## Бонус: LLM-скилл (+2 балла)

Метод `llm_recommend()` отправляет QualityReport в YandexGPT API и получает:
- Анализ обнаруженных проблем
- Рекомендуемую стратегию чистки
- Обоснование выбора

Требуется:
```bash
export YANDEX_API_KEY=...
export YANDEX_FOLDER_ID=...
```

## Структура репозитория

```
hw2-data-quality/
├── agents/
│   └── data_quality_agent.py
├── config.yaml
├── notebooks/
│   └── quality_analysis.ipynb
├── data/
│   ├── raw/dataset.csv
│   └── clean/dataset_clean.csv
├── reports/
│   └── quality_report.md
├── main.py
├── requirements.txt
└── README.md
```

## Требования

Python 3.10+, зависимости: см. requirements.txt
