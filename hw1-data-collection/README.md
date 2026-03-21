# DataCollectionAgent

Агент для автоматического сбора и унификации данных из нескольких источников.
Домашнее задание №1 по курсу AI Agents.

## Задача ML

Sentiment analysis — анализ тональности отзывов на инди-игры Steam.
Агент собирает данные из разных источников (HuggingFace датасет, Steam Reviews API),
приводит их к единой схеме и сохраняет готовый датасет для дальнейшего обучения модели.

## Архитектура агента

```
DataCollectionAgent
│
├── run(sources) ─────────── основной метод
│   ├── _load_dataset()      HuggingFace (ksang/steamreviews)
│   ├── _fetch_steam_reviews() Steam Reviews API по тегу
│   ├── _scrape()            Web scraping (requests + BS4)
│   ├── _fetch_api()         REST API
│   └── _merge()             Объединение + унификация схемы
│
├── get_games_by_tag(tag)    Поиск инди-игр через SteamSpy API
│
└── config.yaml ─────────── конфигурация источников и выхода
```

**Flow:** config.yaml → парсинг источников → сбор данных → merge → CSV

## Источники данных

| # | Тип | Источник | Описание |
|---|-----|----------|----------|
| 1 | HuggingFace | ksang/steamreviews | 6M+ отзывов Steam, берём сэмпл |
| 2 | Steam Reviews API | store.steampowered.com | Отзывы на топ инди-игр по выбранному тегу |

Теги для фильтрации: Indie, Horror, Roguelike, Puzzle, Platformer, RPG, Strategy и др.
Фильтрация AAA-игр через SteamSpy (< 10M владельцев).

## Схема выходных данных

| Колонка | Тип | Описание |
|---------|-----|----------|
| text | str | Текст отзыва |
| label | str | Метка класса (positive/negative) |
| source | str | Идентификатор источника (hf_ksang/steamreviews / steam_{appid}) |
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

# Явный список источников
df = agent.run(sources=[
    {'type': 'hf_dataset', 'name': 'ksang/steamreviews', 'split': 'train', 'sample_size': 500},
    {'type': 'steam_reviews', 'tag': 'Puzzle', 'top_n': 5, 'reviews_per_game': 100},
])

# Поиск игр по тегу
games = DataCollectionAgent.get_games_by_tag('Horror', top_n=10)
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
