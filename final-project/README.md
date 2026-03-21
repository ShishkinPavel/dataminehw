# Final Data Pipeline — ML Portfolio Project

Единый дата-пайплайн с human-in-the-loop, объединяющий всех агентов из заданий 1–4. Реализован как 5 Claude Code скиллов. Агенты импортируются напрямую из оригинальных hw-директорий (без копирования).

## Быстрый старт

```bash
cd dataminehw
pip install -r final-project/requirements.txt

# Запуск всего пайплайна (мастер-скилл):
/pipeline

# Или поэтапно:
/collect          # Шаг 1: сбор данных
/clean            # Шаг 2: чистка
/annotate         # Шаг 3: авторазметка + human review
/active-learn     # Шаг 4: AL + обучение модели
```

## Архитектура: 5 скиллов

| Скилл | Агент | Описание | HITL |
|-------|-------|----------|------|
| `/collect` | DataCollectionAgent (HW1) | HF steam_reviews + Steam API по категории | Спрашивает категорию (Horror, RPG...) и подтверждение игр |
| `/clean` | DataQualityAgent (HW2) | Обнаружение и устранение проблем | Спрашивает стратегию чистки |
| `/annotate` | AnnotationAgent (HW3) | Zero-shot classification + ручная проверка | **Главная HITL-точка**: показывает low-confidence примеры, просит исправить метки |
| `/active-learn` | ActiveLearningAgent (HW4) | AL-цикл + обучение модели | Показывает метрики, спрашивает подтверждение |
| `/pipeline` | Оркестратор | Вызывает 4 скилла последовательно | Все HITL-точки выше |

## Описание задачи

**Задача:** Sentiment Classification — бинарная классификация тональности отзывов на инди-игры Steam

**Модальность:** text (отзывы Steam)

**Классы:** positive, negative

## Что делает каждый агент

### DataCollectionAgent (HW1)
- **Источник 1:** HuggingFace `ksang/steamreviews` — сэмпл из 6M+ отзывов
- **Источник 2:** Steam Reviews API — отзывы на топ инди-игр по выбранной категории (SteamSpy → Steam API)
- Агент спрашивает категорию (Horror, Roguelike, Puzzle...) и показывает найденные игры
- Унифицированная схема: text, label, source, collected_at

### DataQualityAgent (HW2)
- Проверяет данные на пропуски, дубликаты, выбросы, дисбаланс
- Стратегии: drop/mean/median/mode/ffill для пропусков, drop/keep_first для дубликатов, clip_iqr/clip_zscore/drop для выбросов

### AnnotationAgent (HW3)
- Авторазметка через zero-shot classification (facebook/bart-large-mnli)
- Метрики: Cohen's kappa, Agreement %, confidence stats
- Бонус: flag_low_confidence() — HITL фильтрация

### ActiveLearningAgent (HW4)
- Сравнение стратегий: entropy, margin, random (5 итераций, batch_size=20)
- Модель: TF-IDF + LogisticRegression
- Бонус: LLM-анализ стратегий через YandexGPT API

## Human-in-the-Loop

HITL реализован нативно через Claude Code скиллы — агент спрашивает пользователя прямо в диалоге.

**Основная HITL-точка:** `/annotate` — после авторазметки
1. Агент показывает примеры с confidence < 0.7
2. Пользователь исправляет метки прямо в диалоге
3. Исправления сохраняются в `review_queue_corrected.csv`

**Дополнительные HITL-точки:**
- `/collect` — выбор категории и подтверждение списка игр
- `/clean` — выбор стратегии чистки
- `/active-learn` — подтверждение метрик перед сохранением модели

## Выходные артефакты

Все метрики и отчёты генерируются при запуске `/pipeline`:

```
final-project/
├── data/
│   ├── raw/dataset.csv                  # Сырые данные
│   ├── raw/dataset_clean.csv            # После чистки
│   ├── labeled/dataset_labeled.csv      # Авторазметка + HITL
│   ├── labeled/final_dataset.csv        # Финальный датасет
│   ├── labeled/data_card.md             # Описание датасета
│   └── results/al_histories.json        # Метрики AL по итерациям
├── models/sentiment_model.joblib        # Обученная модель
├── plots/
│   ├── learning_curve.png               # Кривая обучения
│   └── strategy_comparison.png          # Сравнение стратегий AL
├── reports/
│   ├── final_report.md                  # Финальный отчёт (5 разделов)
│   ├── quality_report.md                # Отчёт чистки
│   ├── annotation_report.md             # Отчёт авторазметки
│   └── al_report.md                     # Отчёт AL
├── specs/annotation_spec.md             # Спецификация разметки
├── review_queue.csv                     # Очередь на HITL
└── review_queue_corrected.csv           # Исправления человека
```

## Структура репозитория

```
dataminehw/
├── .claude/skills/                  # Claude Code скиллы
│   ├── collect/SKILL.md             # /collect — сбор данных
│   ├── clean/SKILL.md               # /clean — чистка
│   ├── annotate/SKILL.md            # /annotate — разметка + HITL
│   ├── active-learn/SKILL.md        # /active-learn — AL + модель
│   └── pipeline/SKILL.md            # /pipeline — мастер-скилл
│
├── hw1-data-collection/             # HW1: агент сбора данных
│   └── agents/data_collection_agent.py
├── hw2-data-quality/                # HW2: агент чистки данных
│   └── agents/data_quality_agent.py
├── hw3-annotation/                  # HW3: агент авторазметки
│   └── agents/annotation_agent.py
├── hw4-active-learning/             # HW4: агент active learning
│   └── agents/al_agent.py
│
└── final-project/                   # Выходные артефакты
    ├── README.md                    # Этот файл
    ├── requirements.txt
    └── run_pipeline.py              # Standalone скрипт пайплайна
```

## Бонус: LLM в пайплайне (+3 балла)

YandexGPT API используется в ActiveLearningAgent для:
- `llm_explain_selection()` — анализ выбранных примеров
- `llm_recommend_strategy()` — рекомендация лучшей стратегии

```bash
export YANDEX_API_KEY="your-key"
export YANDEX_FOLDER_ID="your-folder-id"
```

## Требования

Python 3.10+. Все зависимости в `final-project/requirements.txt`.
