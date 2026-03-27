# IMDB Movie Reviews — Sentiment Analysis Pipeline

Единый data pipeline для сбора, чистки, разметки и обучения модели sentiment analysis
на отзывах фильмов (IMDB + Rotten Tomatoes). Объединяет 4 агента из HW1-HW4 с human-in-the-loop.

## Быстрый старт

```bash
cd final-project
pip install -r requirements.txt

# Основной запуск (одна команда — требование проекта):
python run_pipeline.py

# С параметрами:
python run_pipeline.py --imdb-size 5000 --rt-size 1000

# Дашборд:
streamlit run dashboard.py
```

## Что делает пайплайн

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ DataCollection    │────>│ DataQuality       │────>│ Annotation        │────>│ ActiveLearning    │
│ Agent (HW1)       │     │ Agent (HW2)       │     │ Agent (HW3)       │     │ Agent (HW4)       │
│                   │     │                   │     │                   │     │                   │
│ HuggingFace       │     │ detect_issues()   │     │ auto_label()      │     │ run_cycle()       │
│ IMDB dataset      │     │ fix(strategy)     │     │ check_quality()   │     │ compare()         │
│                   │     │ compare()         │     │ HITL review       │     │ fit(full)         │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
     dataset.csv ────────> dataset_clean.csv ──> dataset_labeled.csv ──> final_dataset.csv
                                                                           sentiment_model.joblib
```

1. **Сбор данных** — HuggingFace `stanfordnlp/imdb` + Rotten Tomatoes REST API (размер настраивается через параметры)
2. **Чистка** — пропуски, дубликаты, выбросы (HITL: выбор стратегии)
3. **Авторазметка** — BART zero-shot classification + **HITL: ревью disagreements BART vs GT**
4. **Active Learning** — сравнение стратегий (entropy, least_confidence, random)
5. **Обучение модели** — TF-IDF + LogisticRegression
6. **Отчёты** — quality, annotation, AL, финальный отчёт + data card

## Human-in-the-Loop

| # | Этап | Тип | Описание |
|---|------|-----|----------|
| 1 | Чистка | Выбор стратегии | Стратегия для пропусков, дубликатов, выбросов |
| 2 | **Разметка** | **Выбор меток + ревью** | **GT vs BART, ревью disagreements — главная HITL-точка** |
| 3 | AL | Конфигурация | Выбор стратегий |
| 4 | Модель | Подтверждение | Одобрение метрик |

## Результаты (последний прогон: 5000 IMDB + 1000 RT)

- **Датасет**: 5998 reviews (GT: 3505 positive / 2493 negative)
- **Модель**: LogisticRegression + TF-IDF, **Accuracy=0.857, F1=0.857**
- **HITL**: 29 BART disagreements найдено, 4 проверено вручную, 0 меток исправлено (все GT подтверждены)
- **AL**: entropy достигает F1≥0.70 на 70 примерах, random — на 210 (экономия 67%)
- **Подробности**: [`reports/final_report.md`](reports/final_report.md)

## Streamlit Dashboard

```bash
cd final-project && streamlit run dashboard.py
```

7 вкладок: EDA, Data Quality, Annotation, HITL Review, Active Learning, Model, Reports.
HITL Review — интерактивная правка меток через web-интерфейс.

## LLM-бонус (YandexGPT)

```bash
echo "YANDEX_API_KEY=your_key" >> .env
echo "YANDEX_FOLDER_ID=your_folder_id" >> .env
```

YandexGPT используется для:
- Рекомендации стратегии чистки данных (`DataQualityAgent.llm_recommend`)
- Анализа и сравнения AL-стратегий (`ActiveLearningAgent.llm_recommend_strategy`)
- Объяснения выбора информативных примеров (`ActiveLearningAgent.llm_explain_selection`)

## Claude Code Skills (slash-команды)

Пайплайн можно запускать не только через `python run_pipeline.py`, но и через **Claude Code skills** — интерактивные slash-команды, которые оркестрируют агентов с human-in-the-loop.

### Требования

- [Claude Code CLI](https://claude.ai/code) установлен и авторизован
- Python 3.10+ с зависимостями из `requirements.txt`
- Для LLM-бонусов: `YANDEX_API_KEY` и `YANDEX_FOLDER_ID` в `final-project/.env`

### Доступные команды

| Команда | Описание | Аргументы | HITL |
|---------|----------|-----------|------|
| `/collect` | Сбор данных (IMDB + Rotten Tomatoes) | `[размер выборки]`, напр. `6000` | Выбор размера выборки |
| `/clean` | Чистка данных (пропуски, дубли, выбросы) | `[путь к датасету]` | Выбор стратегии чистки |
| `/annotate` | Авторазметка BART + ревью disagreements | `[путь к датасету или кол-во строк]` | Выбор меток, ревью ошибок BART |
| `/active-learn` | Active Learning + обучение модели | `[стратегии]`, напр. `entropy,random` | Выбор стратегий, одобрение модели |
| `/pipeline` | Запуск всех 4 шагов последовательно | `[пожелания]` | Все HITL-точки выше |

### Примеры использования

```bash
# Запуск в Claude Code CLI:

# Полный пайплайн (все шаги последовательно)
/pipeline

# Только сбор данных с указанием размера
/collect 5000

# Чистка конкретного файла
/clean final-project/data/raw/dataset.csv

# Авторазметка
/annotate

# Active Learning с конкретными стратегиями
/active-learn entropy,random,least_confidence
```

### Пайплайн по шагам

```
/collect → /clean → /annotate → /active-learn
   │          │          │            │
   ▼          ▼          ▼            ▼
dataset.csv  dataset     dataset      sentiment_model.joblib
EDA plots    _clean.csv  _labeled.csv AL plots + reports
             quality     annotation   al_report.md
             _report.md  _report.md   final_report.md
```

### HITL-точки (Human-in-the-Loop)

Каждая slash-команда задаёт вопросы через интерактивный интерфейс Claude Code (не текстом). Основные точки:

1. **`/collect`** — выбор размера IMDB (3000/5000/8000) и RT (500/1000/2000)
2. **`/clean`** — стратегия для пропусков (drop/mean/median/mode), дубликатов (drop/keep_first), выбросов (clip_iqr/clip_zscore/drop). Дополнительно: рекомендация YandexGPT
3. **`/annotate`** — главная HITL-точка:
   - Выбор меток для обучения (GT / BART / гибрид)
   - Ручной ревью disagreements BART vs Ground Truth
   - Исправление конкретных меток по одной
4. **`/active-learn`** — запуск дополнительных стратегий, одобрение финальной модели

### Артефакты

После полного прогона `/pipeline` создаются:

| Категория | Файлы |
|-----------|-------|
| Данные | `data/raw/dataset.csv`, `data/raw/dataset_clean.csv`, `data/labeled/final_dataset.csv` |
| Модель | `models/sentiment_model.joblib` |
| Графики | `plots/eda_overview.png`, `plots/learning_curve.png`, `plots/strategy_comparison.png` |
| Отчёты | `reports/quality_report.md`, `reports/annotation_report.md`, `reports/al_report.md`, `reports/final_report.md` |
| HITL | `review_queue.csv`, `review_queue_corrected.csv`, `export/labelstudio_import.json` |
| Data card | `data/labeled/data_card.md` |

### HOW TO: Использование другого датасета

По умолчанию `/collect` загружает IMDB + Rotten Tomatoes, но `DataCollectionAgent` поддерживает 5 типов источников — можно собирать данные откуда угодно.

#### Поддерживаемые источники

| Тип | Описание | Пример |
|-----|----------|--------|
| `hf_dataset` | Любой датасет с HuggingFace (через библиотеку `datasets`) | `imdb`, `yelp_review_full`, `amazon_polarity` |
| `hf_api` | Любой HF-датасет через REST API | `cornell-movie-review-data/rotten_tomatoes` |
| `steam_reviews` | Отзывы из Steam по тегу жанра | `Horror`, `Roguelike`, `Puzzle` |
| `scrape` | Веб-скрапинг по URL + CSS-селектору | Любая страница с отзывами |
| `api` | Произвольный REST API | Любой endpoint, возвращающий JSON |

#### Вариант 1: Через `run_pipeline.py` (Python)

Отредактируйте конфиг источников в `run_pipeline.py`:

```python
sources = [
    {
        'type': 'hf_dataset',
        'name': 'yelp_review_full',    # любой HF-датасет
        'split': 'train',
        'sample_size': 5000,
    },
    {
        'type': 'hf_api',
        'dataset': 'cornell-movie-review-data/rotten_tomatoes',
        'split': 'train',
        'sample_size': 1000,
        'label_map': {0: 'negative', 1: 'positive'},
    },
]
```

#### Вариант 2: Через Claude Code (естественный язык)

Просто опишите что нужно — Claude сформирует конфиг автоматически:

```
# Собрать отзывы из Steam вместо IMDB
/collect собери отзывы на инди-хорроры из Steam, 500 штук

# Использовать другой HF-датасет
/collect используй yelp_review_full вместо imdb, 3000 строк

# Комбинировать источники
/collect imdb 3000 + steam horror 1000
```

#### Вариант 3: Прямой вызов агента (Python-скрипт)

```python
import sys, yaml, tempfile
sys.path.insert(0, 'hw1-data-collection')
from agents.data_collection_agent import DataCollectionAgent

# Steam-отзывы на хорроры
sources = [
    {
        'type': 'steam_reviews',
        'tag': 'Horror',
        'top_n': 5,
        'reviews_per_game': 200,
    },
    {
        'type': 'hf_dataset',
        'name': 'imdb',
        'split': 'train',
        'sample_size': 3000,
    },
]

config = {'sources': sources, 'output': {'path': 'final-project/data/raw/dataset.csv'}}
tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
yaml.dump(config, tmp); tmp.close()

agent = DataCollectionAgent(config=tmp.name)
df = agent.run()
print(f"Собрано {len(df)} строк из {df['source'].nunique()} источников")
```

#### Ограничения

- **Схема данных фиксирована**: любой источник должен отдавать `text` и `label` (или маппинг через `label_map`)
- **Бинарный sentiment**: пайплайн заточен под `positive`/`negative`. Мультикласс (5 звёзд) нужно маппить в бинарные метки
- **BART-аннотация**: работает только для английских текстов (модель `facebook/bart-large-mnli`)
- **Steam API**: rate-limited, большие выборки (>2000) могут занять время

### Важно

- Скиллы используют агентов из `hw1-hw4/` напрямую — это source of truth
- Все отчёты генерируются на русском (технические термины на английском)
- `.csv`, `.png`, `.joblib` в `.gitignore` — это генерируемые артефакты
- `/pipeline` вызывает остальные скиллы последовательно — не нужно запускать их отдельно

## Структура проекта

```
final-project/
├── agents/                          # 4 агента из HW1-HW4
│   ├── data_collection_agent.py
│   ├── data_quality_agent.py
│   ├── annotation_agent.py
│   └── al_agent.py
├── data/
│   ├── raw/                         # Сырые и очищенные данные
│   ├── labeled/                     # Размеченные данные + data card
│   └── results/                     # AL histories
├── models/                          # Обученная модель (.joblib)
├── plots/                           # EDA + AL графики
├── reports/                         # Отчёты (quality, annotation, AL, final)
├── specs/                           # Спецификация разметки
├── export/                          # LabelStudio export
├── dashboard.py                     # Streamlit HITL-дашборд
├── run_pipeline.py                  # Основной пайплайн (python run_pipeline.py)
├── requirements.txt                 # Зависимости
└── README.md
```

## Требования

Python 3.10+. Все зависимости в `requirements.txt`:
pandas, scikit-learn, transformers, torch, datasets, requests, matplotlib, scipy, tqdm, pyyaml, joblib, beautifulsoup4, python-dotenv, streamlit.
