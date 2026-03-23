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
