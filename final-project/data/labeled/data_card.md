# Data Card — Movie Reviews Sentiment Dataset

## Обзор

| Параметр | Значение |
|----------|----------|
| **Всего строк** | 5998 |
| **Источник 1** | HuggingFace IMDB (hf_imdb) — 4998 строк |
| **Источник 2** | HF REST API Rotten Tomatoes (api_rotten_tomatoes) — 1000 строк |
| **Задача** | Binary sentiment classification (positive / negative) |
| **Формат** | CSV (text, label, source, collected_at, predicted_label, confidence) |

## Распределение меток

### Ground Truth (label)

| Класс | Количество | Процент |
|-------|-----------|---------|
| positive | 3505 | 58.4% |
| negative | 2493 | 41.6% |

### BART predicted (predicted_label, на выборке 200 строк)

| Класс | Количество | Процент |
|-------|-----------|---------|
| negative | 102 | 51.0% |
| positive | 98 | 49.0% |

Для остальных 5798 строк BART-предсказание не запускалось (NaN).

### Final labels = GT (без коррекций)

Пользователь проверил 4 disagreements BART vs GT, подтвердил все GT метки. Финальные метки = GT без изменений.

## Язык

- **Основной:** English (100%)
- Все тексты на английском языке (IMDB и Rotten Tomatoes — англоязычные платформы)

## Временной диапазон

- **Дата сбора:** 2026-03-23 (UTC)
- **Период отзывов:** IMDB — разные годы (1940-е — 2020-е), Rotten Tomatoes — аналогично
- **Источники данных актуальны** на момент выгрузки из HuggingFace

## Известные смещения

1. **Selection bias (IMDB):** выборка 5000 из 25000 train, случайная с seed=42. Репрезентативна, но не полна
2. **Selection bias (RT):** 1000 строк из train split (offset-based), могут иметь смещение по порядку
3. **Label bias:** IMDB имеет только бинарные метки (positive/negative), промежуточные оценки (3-4 из 10) исключены из датасета авторами IMDB
4. **BART annotation errors:** на выборке 200 строк Cohen's kappa = 0.711 (хорошее согласие). BART систематически ошибается на positive отзывах с негативной лексикой (ужасы, смешанные рецензии)
5. **Length bias:** IMDB отзывы значительно длиннее RT рецензий, модель может полагаться на длину как прокси

## Конвейер

```
1. DataCollectionAgent  → 6000 строк (IMDB 5000 + RT 1000)
2. DataQualityAgent     → 5998 строк (удалено 2 дубликата)
3. AnnotationAgent      → BART-аудит на 200 строках, 0 HITL-коррекций (GT подтверждены)
4. ActiveLearningAgent  → entropy vs random, F1=0.76 (AL) → 0.857 (full train)
5. Final Model          → LogisticRegression + TF-IDF, accuracy=0.857, F1=0.857
```

## Ограничения

1. **Бинарная классификация** — нет нейтрального класса, смешанные отзывы вынуждены быть positive или negative
2. **Домен** — только кинорецензии, не переносится на другие домены (товары, рестораны)
3. **Язык** — только английский
4. **Temporal** — отзывы за длительный период, стиль языка менялся (фильмы 1940-х vs 2020-х)
5. **Annotation quality** — GT-метки из IMDB не верифицированы вручную (кроме 4 проверенных примеров)
