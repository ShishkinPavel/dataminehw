---
name: annotate
description: "Авторазметка данных через AnnotationAgent (HW3) — zero-shot classification + human review"
argument-hint: "[путь к датасету или количество строк]"
---

# Шаг 3: Авторазметка + Human Review (AnnotationAgent)

Ты выполняешь этап авторазметки и ручной проверки для финального пайплайна.

## Контекст

Источник (HuggingFace IMDB) **уже содержит ground truth метки** (`label`: 0=negative, 1=positive).
BART zero-shot НЕ нужен для разметки — данные уже размечены.
Роль BART здесь:
1. **Демонстрация HW3** — показать работу AnnotationAgent
2. **Quality signal** — сравнить авторазметку с GT, выявить где BART ошибается
3. **HITL** — дать пользователю проверить спорные примеры (disagreements BART vs GT)

**ВАЖНО:** BART добавляет колонки `predicted_label` и `confidence`. Колонка `label` (GT) **никогда не перезаписывается**. Финальная модель учится на GT.

## Что делать

### Часть A: Аудит GT через BART (на выборке 100-200 строк)

1. Импортируй `AnnotationAgent` из `hw3-annotation/agents/annotation_agent.py`
2. Загрузи данные из `final-project/data/raw/dataset_clean.csv`
3. Возьми **стратифицированную выборку** 100-200 строк (сохраняя пропорцию positive/negative)
4. Используй конфиг `hw3-annotation/config.yaml`
5. Запусти `agent.auto_label(sample_df)` — BART на выборке. Дополняет колонками `predicted_label` и `confidence`, не трогая `label`
6. Запусти `agent.check_quality(sample_labeled)` — сравнение BART vs GT
7. Сгенерируй спецификацию: `agent.generate_spec(sample_labeled)`
8. Флагни disagreements: примеры где BART ≠ GT
9. **Export в LabelStudio** (HW3 requirement): вызови `agent.export_to_labelstudio(sample_labeled)`

### Часть B: Анализ качества BART — ОБЯЗАТЕЛЬНО

BART `facebook/bart-large-mnli` систематически ошибается на длинных рецензиях (сложная лексика, обсуждение негативных тем в позитивном ключе).
**Перед HITL-ревью обязательно:**

1. Сравни `predicted_label` (BART) с `label` (ground truth из IMDB/HF).
   Посчитай:
   - Количество disagreements (где BART ≠ ground truth)
   - Разбивку: GT=positive→BART=negative vs GT=negative→BART=positive
   - Cohen's kappa (из `check_quality`)
2. Покажи пользователю **полную картину**:
   - "BART переразметил X positive отзывов в negative (Y% ошибок)"
   - "Cohen's kappa = Z (fair/moderate/good)"
   - Покажи 5 примеров high-confidence ошибок BART (conf>0.9 + disagreement с GT)
3. Используй `AskUserQuestion` (header: "Метки"):
   - "Какие метки использовать для обучения модели?" — варианты:
     - "Ground truth (IMDB label) (Recommended)" — если kappa < 0.5
     - "BART predicted_label" — если kappa > 0.6
     - "Гибрид: GT + BART для неразмеченных"

### Часть C: Human-in-the-Loop Review

Это **главная HITL-точка** всего пайплайна:

1. Покажи пользователю статистику: disagreements BART vs GT (не только low-confidence!)
2. Покажи 10 примеров **disagreements** с самым высоким confidence BART — это самые опасные ошибки
3. Используй `AskUserQuestion` (header: "Ревью"): "Хотите исправить метки вручную?" — варианты: "Да, покажи примеры", "Нет, оставить как есть", "Симулировать (автокоррекция)"
4. Если "Да" — для каждого примера используй `AskUserQuestion` (header: "Метка"):
   - Используй поле `preview` для показа **полного текста отзыва** (не обрезай!). Preview рендерится в отдельном блоке и поддерживает длинный текст с переносами строк.
   - В `description` опции укажи GT label и BART label.
   - Варианты: "positive", "negative", "Оставить как есть". Каждый вариант должен иметь `preview` с полным текстом + пометкой какую метку выбирает пользователь.
   - **НЕ обрезай текст** — пользователь должен видеть весь отзыв целиком чтобы принять решение.
5. Сохрани исправления в `final-project/review_queue_corrected.csv`
6. Примени исправления к датасету
НЕ задавай вопросы текстом — только через AskUserQuestion.

Если пользователь говорит "пропусти" или "симулируй" — используй ground truth метки для автокоррекции.

### Часть D: Формирование final_dataset

**ВАЖНО:** `final_dataset.csv` сохраняет **все три колонки**:
- `label` — ground truth (из IMDB/HF). Это основная метка для обучения.
- `predicted_label` — BART prediction (для анализа, не для обучения)
- `confidence` — уверенность BART

**НИКОГДА не делай `df['label'] = df['predicted_label']`.** GT метки — от человеческих аннотаторов IMDB, они надёжнее BART.

## Агент

Используй `hw3-annotation/agents/annotation_agent.py` напрямую.
HITL-часть делай интерактивно через диалог.

## Оптимизация

Цель BART — **аудит качества GT**, а не разметка датасета. GT метки уже есть.

- Запусти BART на **100-200 примерах** (стратифицированная выборка). Этого достаточно чтобы:
  - Оценить agreement BART vs GT
  - Найти типы ошибок (сложная лексика, негативные темы в позитивном ключе)
  - Показать примеры disagreements для HITL
  - Сделать вывод о качестве GT (обычно: "GT хорошего качества, расхождения на неоднозначных отзывах")
- НЕ гоняй BART на всём датасете — это бессмысленно и долго
- Финальная модель учится на GT метках **всего** датасета

## Терминология в отчётах

Во всех отчётах чётко различай:
- **Ground truth (GT)** — исходные метки из IMDB (`label`) или HuggingFace (`review_score`)
- **BART predicted** — метки от zero-shot классификатора
- **Final label** — метки, выбранные для обучения (GT или BART, по решению пользователя)
Никогда не путай эти три типа меток. Указывай тип метки при каждом упоминании распределения.

## Язык

Все отчёты (`annotation_report.md`, `annotation_spec.md`) — **на русском**. Заголовки разделов на русском: "Спецификация разметки", "Задача", "Классы", "Граничные случаи", "Инструкция для разметчика". Технические термины (Cohen's kappa, LabelStudio, zero-shot) оставлять на английском.

## Выходные файлы

- `final-project/data/labeled/dataset_labeled.csv`
- `final-project/data/labeled/final_dataset.csv` — с колонкой `label` = выбранные метки
- `final-project/reports/annotation_report.md`
- `final-project/specs/annotation_spec.md`
- `final-project/review_queue.csv` (disagreements для HITL)
- `final-project/review_queue_corrected.csv` (исправления)
