---
name: collect
description: "Сбор данных из Steam: HF dataset + Steam Reviews API по категории (HW1)"
disable-model-invocation: true
argument-hint: "[категория: Horror, Roguelike, Puzzle, Platformer, RPG...]"
---

# Шаг 1: Сбор данных — Steam Gaming Reviews

Ты выполняешь этап сбора данных для финального пайплайна.
Задача: sentiment analysis отзывов на инди-игры в Steam.

## Источники

1. **HuggingFace** `ksang/steamreviews` — 6M+ отзывов, берём сэмпл
2. **Steam Reviews API** — отзывы на топ инди-игр по выбранной категории

## Порядок действий

1. **Выбор категории** (если не передана в $ARGUMENTS):
   Используй инструмент `AskUserQuestion` с вариантами выбора:
   - Вопрос 1 (header: "Категория"): Horror, Roguelike, Puzzle, Platformer (+ Other для остальных: RPG, Strategy, Simulation, Survival, Adventure)
   - Вопрос 2 (header: "Кол-во игр"): 3, 5 (Recommended), 10
   - Вопрос 3 (header: "Отзывов/игру"): 50, 100 (Recommended), 200
   НЕ задавай вопросы текстом — только через AskUserQuestion.

2. Импортируй `DataCollectionAgent` из `hw1-data-collection/agents/data_collection_agent.py`

3. Покажи пользователю какие игры нашлись по тегу (вызови `DataCollectionAgent.get_games_by_tag(tag)`)
   - Используй `AskUserQuestion` (header: "Игры"): "Эти игры подходят?" — варианты: "Да, продолжаем", "Нет, выбрать другой тег"

4. Запусти сбор через `agent.run()` с конфигом:
   ```yaml
   sources:
     - type: hf_dataset
       name: ksang/steamreviews
       split: train
       sample_size: 500
     - type: steam_reviews
       tag: <выбранная категория>
       top_n: 5
       reviews_per_game: 100
   ```

5. Сохрани в `final-project/data/raw/dataset.csv`
6. Покажи статистику: количество, источники, распределение меток

## Агент

Используй `hw1-data-collection/agents/data_collection_agent.py` напрямую.

## Выходные файлы

- `final-project/data/raw/dataset.csv` — собранный датасет (text, label, source, collected_at)
