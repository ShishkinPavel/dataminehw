---
name: collect
description: "Сбор данных: IMDB (HF dataset) + Rotten Tomatoes (HF REST API) (HW1)"
argument-hint: "[размер выборки, например 6000]"
---

# Шаг 1: Сбор данных — Movie Reviews (IMDB + Rotten Tomatoes)

Ты выполняешь этап сбора данных для финального пайплайна.
Задача: sentiment analysis отзывов на фильмы.

## Источники (2 — требование HW1)

1. **HuggingFace dataset** `imdb` — 50K отзывов (25K train + 25K test). Основной источник, берём 5000 из train split. Загружается через библиотеку `datasets` (`_load_dataset()`).
2. **HuggingFace Datasets REST API** `cornell-movie-review-data/rotten_tomatoes` — 10K коротких рецензий. Дополнительный источник, берём 1000 через REST API (`_fetch_hf_api()`). Это именно **API-источник** (HTTP GET к `datasets-server.huggingface.co/rows`).

## Порядок действий

1. **Размер выборки** (если не передан в $ARGUMENTS):
   Используй инструмент `AskUserQuestion` с вариантами выбора:
   - Вопрос 1 (header: "Размер IMDB"): 3000, 5000 (Recommended), 8000
   - Вопрос 2 (header: "Размер RT"): 500, 1000 (Recommended), 2000
   НЕ задавай вопросы текстом — только через AskUserQuestion.

2. Импортируй `DataCollectionAgent` из `hw1-data-collection/agents/data_collection_agent.py`

3. Запусти сбор через `agent.run()` с конфигом:
   ```yaml
   sources:
     - type: hf_dataset
       name: imdb
       split: train
       sample_size: 5000
     - type: hf_api
       dataset: cornell-movie-review-data/rotten_tomatoes
       split: train
       sample_size: 1000
       label_map:
         0: negative
         1: positive
   ```

4. Сохрани в `final-project/data/raw/dataset.csv`

5. Покажи статистику: количество, источники, распределение меток.
   **Важно:** IMDB label — целые числа (0/1). Сразу после сбора конвертируй: `0 → 'negative'`, `1 → 'positive'`. Проверь `df['label'].unique()` — должны быть только строки `'positive'`/`'negative'`.

6. **Если RT API вернул меньше строк чем запрошено** — это нормально (ограничение HF Datasets Server). Выведи предупреждение: "RT API вернул N из M строк (ограничение API)". Не пытайся обойти — продолжай с тем что есть.

7. **EDA (обязательно по HW1):**
   - Распределение классов (bar chart)
   - Длины текстов (histogram + median)
   - Топ-20 слов (bar chart)
   - Распределение по источникам (IMDB vs Rotten Tomatoes)
   Сохрани графики в `final-project/plots/eda_overview.png` и `final-project/plots/eda_top_words.png`.

## Агент

Используй `hw1-data-collection/agents/data_collection_agent.py` напрямую.
- `_load_dataset()` — загрузка через HF `datasets` library (источник 1)
- `_fetch_hf_api()` — загрузка через HF Datasets REST API (источник 2)

## Выходные файлы

- `final-project/data/raw/dataset.csv` — собранный датасет (text, label, source, collected_at)
- `final-project/plots/eda_overview.png` — EDA визуализации
- `final-project/plots/eda_top_words.png` — топ-20 слов
