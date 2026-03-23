# Отчёт об авторазметке (HW3)

## Обзор

- **Датасет:** 5998 строк (после чистки)
- **Выборка для BART:** 200 строк (стратифицированная)
- **Модель:** `facebook/bart-large-mnli` (zero-shot classification)
- **Финальные метки для обучения:** Ground truth (GT) из IMDB/HF

## Результаты BART на выборке

### Распределение BART predicted меток
- **positive:** 98 (49.0%)
- **negative:** 102 (51.0%)

### Распределение GT меток (в выборке)
- **positive:** 117 (58.5%)
- **negative:** 83 (41.5%)

### Метрики согласия

| Метрика | Значение |
|---------|---------|
| Cohen's kappa | 0.711 (хорошее согласие) |
| Agreement | 85.5% |
| Disagreements | 29 из 200 (14.5%) |

### Classification Report (BART vs GT)

```
              precision    recall  f1-score   support
    negative       0.76      0.94      0.84        83
    positive       0.95      0.79      0.87       117
    accuracy                           0.85       200
   macro avg       0.86      0.87      0.85       200
weighted avg       0.87      0.85      0.86       200
```

### Уверенность модели
- **Средняя:** 0.8824
- **Std:** 0.1347
- **Ниже порога 0.7:** 25 примеров (12.5%)

## Анализ ошибок BART

### Направление ошибок
- **GT=positive → BART=negative:** 24 случая (82.8% ошибок)
- **GT=negative → BART=positive:** 5 случаев (17.2% ошибок)

### Типичные паттерны ошибок BART

1. **Негативная лексика в позитивном контексте:** Отзывы обсуждающие ужасы, насилие, скандальные темы — BART реагирует на слова, а не на тон. Пример: отзыв о "Tarzan and His Mate" обсуждает цензуру ('immoral', 'sin'), но автор восхищается фильмом.

2. **Смешанные отзывы с позитивным итогом:** Автор критикует, но ставит положительную оценку (6.5/10). BART не улавливает итоговый вердикт.

3. **Сарказм и ирония:** Отзыв начинается с "Wow...sheer brilliance" — BART принимает за похвалу, но это сарказм (GT=negative).

## HITL-ревью

### Проверено пользователем: 4 примера

| # | GT | BART | Conf | Решение | Текст (начало) |
|---|-----|------|------|---------|----------------|
| 1 | positive | negative | 0.982 | **positive** (GT подтверждён) | "As much as I have enjoyed the Hanzo..." |
| 2 | negative | positive | 0.981 | **negative** (GT подтверждён) | "Actually, this flick, made in 1999..." |
| 3 | positive | negative | 0.965 | **positive** (GT подтверждён) | "Hard to believe, perhaps, but this film..." |
| 4 | positive | negative | 0.955 | **positive** (GT подтверждён) | "While movie titles contains the word Mother..." |

**Результат:** Все GT метки подтверждены. BART ошибался на всех 4 проверенных примерах. Коррекций GT не потребовалось.

### Решение по меткам
Пользователь выбрал **Ground truth (GT)** для обучения модели. Обоснование: Cohen's kappa = 0.71, BART систематически переразмечает positive отзывы в negative при наличии критической лексики.

## Финальный датасет

- **Файл:** `final-project/data/labeled/final_dataset.csv`
- **Строк:** 5998
- **GT: 3505 positive / 2493 negative**
- **Метки для обучения:** Ground truth (колонка `label`)
- **BART predictions:** доступны для 200 строк (колонки `predicted_label`, `confidence`)

## Артефакты

- `dataset_labeled.csv` — выборка 200 строк с BART predictions
- `final_dataset.csv` — полный датасет с GT метками
- `review_queue.csv` — 29 disagreements для ревью
- `review_queue_corrected.csv` — 4 проверенных примера
- `export/labelstudio_import.json` — экспорт в LabelStudio
- `specs/annotation_spec.md` — спецификация разметки
