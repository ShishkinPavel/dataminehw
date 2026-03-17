# DataCollectionAgent

Агент для автоматического сбора и унификации данных из нескольких источников.
Домашнее задание №1 по курсу AI Agents.

## Задача ML

Sentiment analysis — анализ тональности текстов.
Агент собирает данные из разных источников (датасеты, web scraping, API),
приводит их к единой схеме и сохраняет готовый датасет для дальнейшего обучения модели.

## Архитектура агента

```
DataCollectionAgent
│
├── run(sources) ─────────── основной метод
│   ├── _load_dataset()      HuggingFace / Kaggle
│   ├── _scrape()            Web scraping (requests + BS4)
│   ├── _fetch_api()         REST API
│   └── _merge()             Объединение + унификация схемы
│
└── config.yaml ─────────── конфигурация источников и выхода
```

**Flow:** config.yaml → парсинг источников → сбор данных → merge → CSV

## Источники данных

| # | Тип | Источник | Описание | Кол-во |
|---|-----|----------|----------|--------|
| 1 | HuggingFace | imdb | Отзывы на фильмы (pos/neg) | 1000 |
| 2 | Web scraping | quotes.toscrape.com | Цитаты известных людей | 10 |

Итого: **1010 записей** из 2 источников.

## Схема выходных данных

| Колонка | Тип | Описание |
|---------|-----|----------|
| text | str | Текстовое содержимое |
| label | str | Метка класса (positive/negative/quote) |
| source | str | Идентификатор источника (hf_imdb / scrape_quotes) |
| collected_at | datetime | Таймстемп сбора данных (UTC) |

## Быстрый старт

### Установка
```bash
git clone <repo-url>
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

# Явный список источников
df = agent.run(sources=[
    {'type': 'hf_dataset', 'name': 'imdb', 'sample_size': 500},
    {'type': 'scrape', 'url': 'https://quotes.toscrape.com/', 'selector': 'div.quote span.text'},
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
