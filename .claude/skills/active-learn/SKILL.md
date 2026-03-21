---
name: active-learn
description: "Active Learning цикл через ActiveLearningAgent (HW4) — сравнение стратегий + обучение модели"
disable-model-invocation: true
argument-hint: "[стратегии: entropy,random,margin]"
---

# Шаг 4: Active Learning + Обучение модели (ActiveLearningAgent)

Ты выполняешь этап Active Learning и обучения финальной модели.

## Что делать

### Часть A: Active Learning

1. Импортируй `ActiveLearningAgent` из `hw4-active-learning/agents/al_agent.py`
2. Загрузи размеченные данные из `final-project/data/labeled/dataset_labeled.csv`
3. Раздели на labeled (50), pool, test
4. Запусти AL-циклы для стратегий (по умолчанию: entropy, random, least_confidence)
   ВАЖНО: entropy и margin дают идентичные результаты при 2 классах (математически эквивалентны).
   Используй least_confidence вместо margin как третью стратегию по умолчанию.
5. Сгенерируй графики: `agent.report()`, `agent.compare_strategies()`
6. Сохрани результаты в `final-project/data/results/al_histories.json`
7. Сохрани отчёт в `final-project/reports/al_report.md`

### Часть B: Обучение финальной модели

1. Обучи LogisticRegression на полном размеченном датасете (TF-IDF + clf)
2. Выведи accuracy, F1, classification report
3. Сохрани модель в `final-project/models/sentiment_model.joblib`
4. Сохрани финальный датасет в `final-project/data/labeled/final_dataset.csv`

### Часть C: LLM-бонус (YandexGPT)

Если настроены YANDEX_API_KEY и YANDEX_FOLDER_ID:
- Вызови `agent.llm_recommend_strategy(histories)` — анализ стратегий
- Покажи результат пользователю

## Агент

Используй `hw4-active-learning/agents/al_agent.py` напрямую.

## HITL

- После сравнения стратегий используй `AskUserQuestion` (header: "Стратегии"): "Запустить дополнительные стратегии?" — варианты: "Нет, достаточно", "Да, добавить margin", "Да, добавить least_confidence"
- Перед сохранением модели покажи метрики и используй `AskUserQuestion` (header: "Модель"): "Сохранить модель с этими метриками?" — варианты: "Да, сохранить", "Нет, перезапустить с другими параметрами"
НЕ задавай вопросы текстом — только через AskUserQuestion.

## Выходные файлы

- `final-project/plots/learning_curve.png`
- `final-project/plots/strategy_comparison.png`
- `final-project/models/sentiment_model.joblib`
- `final-project/reports/al_report.md`
- `final-project/data/results/al_histories.json`
- `final-project/data/labeled/final_dataset.csv`
