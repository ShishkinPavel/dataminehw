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
- Задача: sentiment analysis отзывов на инди-игры Steam, обучить модель, написать отчёт

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
**HITL**: покажи low-confidence примеры, дай пользователю исправить метки.
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

## Правила

- Между шагами показывай пользователю прогресс (что сделано, что дальше)
- Не пропускай HITL-точки — спрашивай пользователя
- Если пользователь просит пропустить шаг или изменить параметры — адаптируйся
- Если $ARGUMENTS содержит пожелания — учти их на всех этапах

## Финальные артефакты

Проверь что все файлы созданы:
- `final-project/data/raw/dataset.csv`
- `final-project/data/raw/dataset_clean.csv`
- `final-project/data/labeled/dataset_labeled.csv`
- `final-project/data/labeled/final_dataset.csv`
- `final-project/data/labeled/data_card.md`
- `final-project/data/results/al_histories.json`
- `final-project/models/sentiment_model.joblib`
- `final-project/plots/strategy_comparison.png`
- `final-project/reports/final_report.md`
- `final-project/reports/quality_report.md`
- `final-project/reports/annotation_report.md`
- `final-project/reports/al_report.md`
- `final-project/specs/annotation_spec.md`
- `final-project/review_queue.csv`
