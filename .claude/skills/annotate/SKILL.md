---
name: annotate
description: "Авторазметка данных через AnnotationAgent (HW3) — zero-shot classification + human review"
disable-model-invocation: true
argument-hint: "[путь к датасету или количество строк]"
---

# Шаг 3: Авторазметка + Human Review (AnnotationAgent)

Ты выполняешь этап авторазметки и ручной проверки для финального пайплайна.

## Что делать

### Часть A: Авторазметка

1. Импортируй `AnnotationAgent` из `hw3-annotation/agents/annotation_agent.py`
2. Загрузи данные из `final-project/data/raw/dataset_clean.csv`
3. Используй конфиг `hw3-annotation/config.yaml`
5. Запусти `agent.auto_label(df)` — zero-shot classification через BART
6. Запусти `agent.check_quality(df_labeled)` — оценка качества
7. Сгенерируй спецификацию: `agent.generate_spec(df_labeled)`
8. Флагни low-confidence: `agent.flag_low_confidence(df_labeled, threshold=0.7)`

### Часть B: Human-in-the-Loop Review

Это **главная HITL-точка** всего пайплайна:

1. Покажи пользователю статистику: сколько примеров с low confidence
2. Покажи 5-10 примеров с самым низким confidence
3. Используй `AskUserQuestion` (header: "Ревью"): "Хотите исправить метки вручную?" — варианты: "Да, покажи примеры", "Нет, оставить как есть", "Симулировать (автокоррекция)"
4. Если "Да" — для каждого примера используй `AskUserQuestion` (header: "Метка"): покажи текст в description, варианты: "positive", "negative", "Оставить как есть"
5. Сохрани исправления в `final-project/review_queue_corrected.csv`
6. Примени исправления к датасету
НЕ задавай вопросы текстом — только через AskUserQuestion.

Если пользователь говорит "пропусти" или "симулируй" — используй ground truth метки для автокоррекции.

## Агент

Используй `hw3-annotation/agents/annotation_agent.py` напрямую.
HITL-часть делай интерактивно через диалог.

## Оптимизация

- Если пользователь просит разметить только часть данных — сделай сэмпл
- Для экономии времени можно разметить 100-200 строк вместо всего датасета

## Выходные файлы

- `final-project/data/labeled/dataset_labeled.csv`
- `final-project/reports/annotation_report.md`
- `final-project/specs/annotation_spec.md`
- `final-project/review_queue.csv` (для HITL)
- `final-project/review_queue_corrected.csv` (исправления)
