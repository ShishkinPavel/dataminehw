# Steam Indie Reviews — Sentiment Analysis Pipeline

Единый data pipeline для сбора, чистки, разметки и обучения модели sentiment analysis
на отзывах инди-игр Steam. Объединяет 4 агента из HW1-HW4 с human-in-the-loop.

## Быстрый старт

```bash
cd final-project
pip install -r requirements.txt
python run_pipeline.py --tag RPG --top-n 5 --reviews 100
```

### Варианты запуска

```bash
# Standalone Python-скрипт (воспроизводимый):
python run_pipeline.py --tag RPG --top-n 10 --reviews 100

# Через Claude Code скиллы (интерактивный):
/pipeline
# или поэтапно: /collect → /clean → /annotate → /active-learn
```

## Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--tag` | Horror | Steam-тег (RPG, Horror, Roguelike, Puzzle, Platformer) |
| `--top-n` | 5 | Количество инди-игр |
| `--reviews` | 100 | Отзывов на игру |
| `--hf-sample` | 500 | Сэмпл из HuggingFace |

## Что делает пайплайн

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ DataCollection    │────>│ DataQuality       │────>│ Annotation        │────>│ ActiveLearning    │
│ Agent (HW1)       │     │ Agent (HW2)       │     │ Agent (HW3)       │     │ Agent (HW4)       │
│                   │     │                   │     │                   │     │                   │
│ HuggingFace +     │     │ detect_issues()   │     │ auto_label()      │     │ run_cycle()       │
│ Steam API         │     │ fix(strategy)     │     │ flag_low_conf()   │     │ compare()         │
│                   │     │ compare()         │     │ HITL review       │     │ fit(full)         │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
     dataset.csv ────────> dataset_clean.csv ──> dataset_labeled.csv ──> final_dataset.csv
                                                                           sentiment_model.joblib
```

1. **Сбор данных** — HuggingFace `ksang/steamreviews` + Steam Reviews API
2. **Чистка** — пропуски, дубликаты, выбросы (HITL: выбор стратегии)
3. **Авторазметка** — zero-shot classification (BART) + **HITL: ревью low-confidence**
4. **Active Learning** — сравнение стратегий (entropy, margin, random)
5. **Обучение модели** — TF-IDF + LogReg/RandomForest
6. **Отчёты** — quality, annotation, AL, финальный отчёт + data card

## Human-in-the-Loop

| # | Этап | Тип | Описание |
|---|------|-----|----------|
| 1 | Сбор | Конфигурация | Выбор категории и подтверждение игр |
| 2 | Чистка | Выбор стратегии | Стратегия для пропусков, дубликатов, выбросов |
| 3 | **Разметка** | **Коррекция меток** | **Ревью low-confidence — главная HITL-точка** |
| 4 | AL | Конфигурация | Выбор стратегий |
| 5 | Модель | Подтверждение | Одобрение метрик |

Варианты HITL-ревью в `run_pipeline.py`:
1. **Интерактивный** — проверять примеры один за другим (p/n/Enter)
2. **Файловый** — заполнить `review_queue.csv` вручную
3. **Автокоррекция** — использовать ground truth (для демо)
4. **Пропустить** — оставить автометки

## LLM-бонус (YandexGPT)

```bash
# Создайте .env в корне final-project:
echo "YANDEX_API_KEY=your_key" >> .env
echo "YANDEX_FOLDER_ID=your_folder_id" >> .env
```

YandexGPT используется для:
- Рекомендации стратегии чистки данных (`DataQualityAgent.llm_recommend`)
- Анализа и сравнения AL-стратегий (`ActiveLearningAgent.llm_recommend_strategy`)
- Объяснения выбора информативных примеров (`ActiveLearningAgent.llm_explain_selection`)

## Результаты

- **Датасет**: 1310 размеченных отзывов (positive/negative)
- **Модель**: RandomForest + TF-IDF, **accuracy=0.76, F1=0.75**
- **HITL**: 10 примеров проверено, 3 метки исправлены
- **AL**: entropy vs random — random стабильнее при шумных метках
- **Подробности**: [`reports/final_report.md`](reports/final_report.md)

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
├── plots/                           # Графики AL
├── reports/                         # Отчёты (quality, annotation, AL, final)
├── specs/                           # Спецификация разметки
├── run_pipeline.py                  # Точка входа
├── config_al.yaml                   # Конфиг Active Learning
├── config_annotation.yaml           # Конфиг разметки
├── requirements.txt                 # Зависимости
├── review_queue.csv                 # Очередь на HITL-ревью
└── README.md
```

## Требования

Python 3.10+. Все зависимости в `requirements.txt`:
pandas, scikit-learn, transformers, torch, datasets, requests, matplotlib, scipy, tqdm, pyyaml, joblib, beautifulsoup4, python-dotenv.
