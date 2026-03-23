# DataCollectionAgent

Агент для автоматического сбора и унификации данных из нескольких источников.
Домашнее задание №1 по курсу AI Agents.

## Задача ML

Sentiment analysis — анализ тональности отзывов на фильмы (IMDB + Rotten Tomatoes).
Агент собирает данные из двух источников, приводит их к единой схеме и сохраняет готовый датасет для дальнейшего обучения модели.

## Архитектура агента

```
DataCollectionAgent
│
├── run(sources) ─────────── основной метод
│   ├── _load_dataset()      HuggingFace library (imdb)
│   ├── _fetch_hf_api()      HuggingFace Datasets REST API (rotten_tomatoes)
│   ├── _scrape()            Web scraping (requests + BS4)
│   ├── _fetch_api()         Generic REST API
│   └── _merge()             Объединение + унификация схемы
│
└── config.yaml ─────────── конфигурация источников и выхода
```

**Flow:** config.yaml → парсинг источников → сбор данных → merge → CSV

## Источники данных

| # | Тип | Источник | Описание |
|---|-----|----------|----------|
| 1 | HF dataset | stanfordnlp/imdb | 50K отзывов, загрузка через `datasets` library |
| 2 | REST API | cornell-movie-review-data/rotten_tomatoes | 10K рецензий, загрузка через HF Datasets REST API |

## Схема выходных данных

| Колонка | Тип | Описание |
|---------|-----|----------|
| text | str | Текст отзыва |
| label | str | Метка класса (positive/negative) |
| source | str | Идентификатор источника (hf_imdb / api_rotten_tomatoes) |
| collected_at | datetime | Таймстемп сбора данных (UTC) |

## Быстрый старт

### Установка
```bash
cd hw1-data-collection
pip install -r requirements.txt
```

### Запуск сбора данных
```bash
python main.py
```

Результат: `data/raw/dataset.csv`

### Запуск EDA
```bash
jupyter notebook notebooks/eda.ipynb
```

## Использование в коде

```python
from agents.data_collection_agent import DataCollectionAgent

# Из конфига
agent = DataCollectionAgent(config='config.yaml')
df = agent.run()

# Явный список источников (2 — требование HW1)
df = agent.run(sources=[
    {'type': 'hf_dataset', 'name': 'imdb', 'split': 'train', 'sample_size': 5000},
    {'type': 'hf_api', 'dataset': 'cornell-movie-review-data/rotten_tomatoes',
     'split': 'train', 'sample_size': 1000, 'label_map': {0: 'negative', 1: 'positive'}},
])
```

## EDA

Ноутбук `notebooks/eda.ipynb` содержит:
- Распределение записей по источникам
- Распределение классов (общее и по источникам)
- Гистограмма длин текстов
- Топ-20 слов (без стоп-слов)
- Word clouds

## Структура репозитория

```
hw1-data-collection/
├── agents/
│   └── data_collection_agent.py   # Класс агента
├── config.yaml                     # Конфигурация источников
├── notebooks/
│   └── eda.ipynb                   # EDA и визуализации
├── data/
│   └── raw/
│       └── dataset.csv             # Собранные данные (генерируется)
├── main.py                         # Точка входа
├── requirements.txt                # Зависимости
└── README.md
```

## Требования

- Python 3.10+
- Зависимости: см. requirements.txt
