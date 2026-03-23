---
name: pipeline
description: "Финальный пайплайн — запуск всех 4 агентов последовательно с human-in-the-loop"
disable-model-invocation: true
argument-hint: "[тема датасета или пожелания]"
---

# Финальный Data Pipeline

Ты — оркестратор финального пайплайна. Запускаешь 4 скилла последовательно, с human-in-the-loop на ключевых этапах.

## Общий контекст

- Корень проекта: `dataminehw/`
- Агенты лежат в: `hw1-data-collection/`, `hw2-data-quality/`, `hw3-annotation/`, `hw4-active-learning/`
- Все выходные артефакты — в `final-project/`
- Задача: sentiment analysis отзывов на фильмы (IMDB), обучить модель, написать отчёт

## Порядок выполнения

### Шаг 1: Сбор данных
Вызови скилл `/collect` с пожеланиями пользователя.
Результат: `final-project/data/raw/dataset.csv`

### Шаг 2: Чистка данных
Вызови скилл `/clean`.
**HITL**: покажи проблемы, дай выбрать стратегию чистки через AskUserQuestion.
Результат: `final-project/data/raw/dataset_clean.csv`

### Шаг 3: Авторазметка + Human Review
Вызови скилл `/annotate`.
**HITL**: покажи disagreements BART vs GT, дай пользователю проверить спорные метки.
Это главная HITL-точка пайплайна.
Результат: `final-project/data/labeled/dataset_labeled.csv`

### Шаг 4: Active Learning + Модель
Вызови скилл `/active-learn`.
Результат: модель, графики, отчёт.

### Шаг 5: Финальный отчёт
Сгенерируй `final-project/reports/final_report.md` с 5 разделами:
1. Описание задачи и датасета
2. Что делал каждый агент
3. Описание HITL-точки (сколько проверено, что исправлено)
4. Метрики качества на каждом этапе + итоговые метрики модели
5. Ретроспектива

Сгенерируй `final-project/data/labeled/data_card.md` с секциями:
- Overview (total, labels, sources, games)
- Label Distribution
- Language (English + возможные другие языки)
- Temporal Range (когда собраны данные)
- Known Biases (selection bias, BART annotation errors)
- Pipeline (этапы обработки)
- Limitations

## Правила

- Между шагами показывай пользователю прогресс (что сделано, что дальше)
- Не пропускай HITL-точки — спрашивай пользователя
- Если пользователь просит пропустить шаг или изменить параметры — адаптируйся
- Если $ARGUMENTS содержит пожелания — учти их на всех этапах

## Язык отчётов — ОБЯЗАТЕЛЬНО

**Все отчёты, data card и спецификации пиши на русском языке.** Технические термины (F1, accuracy, Cohen's kappa, TF-IDF и т.д.) можно оставлять на английском, но описания, выводы, обоснования — на русском. Задание на русском → отчёты на русском.

**Заголовки разделов в markdown тоже на русском:** "Обзор" (не "Overview"), "Распределение меток" (не "Label Distribution"), "Известные смещения" (не "Known Biases"), "Ограничения" (не "Limitations"), "Язык" (не "Language"), "Конвейер" (не "Pipeline"). Единственное исключение — устоявшиеся термины в заголовках: "Classification Report", "Cohen's kappa".

## Числа в Data Card — ОБЯЗАТЕЛЬНО

В `data_card.md` все числа (количество строк по источникам, распределение меток) должны быть **вычислены из реальных данных** (`final_dataset.csv`), а не вписаны руками. Посчитай `df.groupby('source').size()` и `df['label'].value_counts()` и вставь реальные значения.

## LLM-бонусы — ОБЯЗАТЕЛЬНО

YandexGPT настроен (ключи в `final-project/.env`). На шагах 2 и 4 скиллы `/clean` и `/active-learn` **обязаны** вызвать LLM-методы агентов. Не пропускай эти вызовы.

## Терминология меток (ОБЯЗАТЕЛЬНО)

Во всех отчётах и выводах чётко различай три типа меток:
- **Ground truth (GT)** — метки из источника (IMDB `label`, HF `review_score`)
- **BART predicted** — метки от zero-shot классификатора
- **Final label** — метки, выбранные для обучения модели (решение принимает пользователь на шаге 3)

При каждом упоминании распределения меток указывай тип. Не пиши просто "990 positive / 422 negative" — пиши "BART predicted: 990 positive / 422 negative" или "GT: 1280 positive / 132 negative".

## Консистентность чисел

После каждого шага выводи единую таблицу состояния:
```
Строк: N | GT: X pos / Y neg | BART: A pos / B neg | Final: ...
```
Это предотвращает путаницу между отчётами.

## Финальные артефакты

Проверь что все файлы созданы:

### Данные
- `final-project/data/raw/dataset.csv`
- `final-project/data/raw/dataset_clean.csv`
- `final-project/data/labeled/dataset_labeled.csv`
- `final-project/data/labeled/final_dataset.csv`
- `final-project/data/labeled/data_card.md`
- `final-project/data/results/al_histories.json`

### Модель и графики
- `final-project/models/sentiment_model.joblib`
- `final-project/plots/eda_overview.png`
- `final-project/plots/eda_top_words.png`
- `final-project/plots/strategy_comparison.png`
- `final-project/plots/learning_curve.png`

### Отчёты и спецификации
- `final-project/reports/final_report.md`
- `final-project/reports/quality_report.md`
- `final-project/reports/annotation_report.md`
- `final-project/reports/al_report.md`
- `final-project/specs/annotation_spec.md`

### HITL и экспорт
- `final-project/review_queue.csv`
- `final-project/review_queue_corrected.csv`
- `final-project/export/labelstudio_import.json`

### Дашборд (бонус +2)
- `final-project/dashboard.py` — Streamlit HITL-дашборд
- Запуск: `cd final-project && streamlit run dashboard.py`
- Вкладки: EDA, Data Quality, Annotation, HITL Review, Active Learning, Model, Reports
- HITL Review — интерактивная правка меток через web-интерфейс

### Шаг 6: Обновление README (после всех шагов)

После генерации всех артефактов **обязательно** перезапиши `final-project/README.md` с актуальными числами из текущего прогона:
- Размер датасета (строки, источники)
- GT распределение меток
- Метрики модели (accuracy, F1)
- Количество HITL-коррекций
- Результаты AL (entropy vs random)

Числа бери из `final_dataset.csv` и `al_histories.json`, не из памяти — чтобы README всегда соответствовал реальным данным.
