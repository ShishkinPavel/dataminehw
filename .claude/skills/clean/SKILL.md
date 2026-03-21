---
name: clean
description: "Чистка данных через DataQualityAgent (HW2) — обнаружение и устранение проблем качества"
disable-model-invocation: true
argument-hint: "[путь к датасету]"
---

# Шаг 2: Чистка данных (DataQualityAgent)

Ты выполняешь этап чистки данных для финального пайплайна.

## Что делать

1. Импортируй `DataQualityAgent` из `hw2-data-quality/agents/data_quality_agent.py`
2. Загрузи данные из `final-project/data/raw/dataset.csv` (или из $ARGUMENTS)
3. Запусти `agent.detect_issues(df)` — покажи отчёт пользователю
4. Примени стратегию чистки через `agent.fix(df, strategy=...)`
5. Сравни до/после через `agent.compare(df_before, df_after)`
6. Сохрани чистые данные в `final-project/data/raw/dataset_clean.csv`
7. Сохрани отчёт в `final-project/reports/quality_report.md`

## Стратегии чистки

- missing: `drop` | `mean` | `median` | `mode` | `ffill`
- duplicates: `drop` | `keep_first` | `keep_last`
- outliers: `clip_iqr` | `clip_zscore` | `drop`

## Агент

Используй `hw2-data-quality/agents/data_quality_agent.py` напрямую.

## HITL — обязательно

После `detect_issues()`:
1. Покажи найденные проблемы (пропуски, дубликаты, выбросы, дисбаланс)
2. Используй `AskUserQuestion` для выбора стратегий:
   - Вопрос 1 (header: "Пропуски"): "Как обработать пропуски?" — drop, mean, median, mode (Recommended), ffill
   - Вопрос 2 (header: "Дубликаты"): "Как обработать дубликаты?" — drop (Recommended), keep_first, keep_last
   - Вопрос 3 (header: "Выбросы"): "Как обработать выбросы?" — clip_iqr (Recommended), clip_zscore, drop
   НЕ задавай вопросы текстом — только через AskUserQuestion.
3. Если пользователь не уверен — предложи рекомендацию (можно через `agent.llm_recommend()` если настроен YandexGPT)
4. После чистки покажи сравнение before/after

Не применяй стратегию молча — это ключевая HITL-точка.
