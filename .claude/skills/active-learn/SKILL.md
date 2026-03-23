---
name: active-learn
description: "Active Learning цикл через ActiveLearningAgent (HW4) — сравнение стратегий + обучение модели"
argument-hint: "[стратегии: entropy,random,margin]"
---

# Шаг 4: Active Learning + Обучение модели (ActiveLearningAgent)

Ты выполняешь этап Active Learning и обучения финальной модели.

## Что делать

### Часть A: Подготовка данных

1. Импортируй `ActiveLearningAgent` из `hw4-active-learning/agents/al_agent.py`
2. Загрузи данные из `final-project/data/labeled/final_dataset.csv`
3. **Используй колонку `label`** — это должны быть метки, выбранные на этапе annotation (GT или BART, по решению пользователя). НЕ перезаписывай `label` на `predicted_label`.
4. Раздели на initial labeled (50), pool, test (20%)
5. **Сбалансируй initial set**: 25 positive + 25 negative (или пропорционально, если negative мало). Случайная выборка из imbalanced данных даёт плохой старт AL.

### Часть B: Active Learning

1. Запусти AL-циклы для стратегий: entropy, random, least_confidence
   **ВАЖНО:**
   - entropy и margin математически эквивалентны при 2 классах — запускай margin только если пользователь попросит, и предупреди об эквивалентности
   - least_confidence ≈ entropy при 2 классах, но может отличаться. Запускай для сравнения
   - При шумных метках (BART) AL может выбирать шумные примеры — высокая entropy означает "модель не уверена", что может значить "метка неправильная", а не "пример информативный"
2. Сгенерируй графики: `agent.report()`, `agent.compare_strategies()`
3. Сохрани результаты в `final-project/data/results/al_histories.json`

### Часть C: Обучение финальной модели

1. Обучи LogisticRegression на **полном** train set (TF-IDF + clf, class_weight='balanced')
2. Выведи accuracy, F1, classification report
3. **Проанализируй negative class.** Если negative precision или recall < 0.5 — это скорее всего из-за малого количества negative примеров, а не плохой модели. Отметь это в отчёте.
4. Сохрани модель в `final-project/models/sentiment_model.joblib`

### Часть D: AL Report

Сохрани отчёт в `final-project/reports/al_report.md`.
В отчёте ОБЯЗАТЕЛЬНО укажи:
- Какие метки использовались (GT или BART) — и почему это важно
- Разницу между стратегиями интерпретируй честно: если разница < 2% F1, это в пределах шума
- Если кривые нестабильны (нет монотонного роста) — отметь это
- **Savings analysis (HW4 requirement):** посчитай при каком N best стратегия достигает F1 random'а на 250 примерах. Разница — "сэкономленные примеры". Например: "entropy достигает F1=0.72 на 130 примерах, random — на 250. Savings: 120 примеров (48%)"

### Часть E: LLM-анализ (YandexGPT) — ОБЯЗАТЕЛЬНО

YandexGPT настроен (ключи в `final-project/.env`). Загрузи `.env` через `dotenv.load_dotenv('final-project/.env')` перед вызовом.

1. **ОБЯЗАТЕЛЬНО** вызови `agent.llm_recommend_strategy(histories)` — анализ и сравнение стратегий от LLM.
   Покажи результат пользователю.
2. **ОБЯЗАТЕЛЬНО** вызови `agent.llm_explain_selection(selected_texts, strategy, iteration)` хотя бы для одной итерации entropy — объяснение почему выбранные примеры информативны.
   Покажи результат пользователю.
3. Включи результаты LLM-анализа в `al_report.md` отдельной секцией "## LLM-анализ (YandexGPT)".
   Если LLM вернёт ошибку — покажи её пользователю и запиши в отчёт, но продолжай работу.

## Агент

Используй `hw4-active-learning/agents/al_agent.py` напрямую.

## HITL

- После сравнения стратегий используй `AskUserQuestion` (header: "Стратегии"): "Запустить дополнительные стратегии?" — варианты: "Нет, достаточно", "Да, добавить margin (≡entropy при 2 классах)"
- Перед сохранением модели покажи метрики и используй `AskUserQuestion` (header: "Модель"): "Сохранить модель с этими метриками?" — варианты: "Да, сохранить", "Нет, перезапустить с другими параметрами"
НЕ задавай вопросы текстом — только через AskUserQuestion.

## Выходные файлы

- `final-project/plots/learning_curve.png`
- `final-project/plots/strategy_comparison.png`
- `final-project/models/sentiment_model.joblib`
- `final-project/reports/al_report.md`
- `final-project/data/results/al_histories.json`
- `final-project/data/labeled/final_dataset.csv`
