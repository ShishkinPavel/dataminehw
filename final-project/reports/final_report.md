# Final Report: Sentiment Analysis of Steam RPG Reviews

## 1. Описание задачи и датасета

**Задача**: бинарная классификация тональности (positive/negative) отзывов на инди-игры Steam в жанре RPG.

**Датасет**:
- 1400 отзывов из двух источников: HuggingFace `ksang/steamreviews` (500) и Steam Reviews API (900)
- 10 инди-RPG игр: Castle Crashers, Robocraft, Kathy Rain, Vampire Survivors, Clicker Heroes, Hades, Undertale, Slay the Spire, Dead Cells
- После чистки: 1311 строк
- Распределение: 908 positive / 403 negative (сильный дисбаланс, ratio=0.13)

## 2. Что делал каждый агент

### DataCollectionAgent (HW1)
- Загрузил 500 отзывов из HuggingFace dataset `ksang/steamreviews`
- Собрал 900 отзывов через Steam Reviews API (10 RPG-игр x 100 отзывов)
- Унифицировал схему: text, label, source, collected_at
- Итого: **1400 строк** в `dataset.csv`

### DataQualityAgent (HW2)
- Обнаружил проблемы: 3 пропуска, 86 дубликатов (6.14%), дисбаланс классов
- Применил стратегию: drop missing, drop duplicates, clip_iqr outliers
- Результат: **1311 строк** (удалено 89 строк)

### AnnotationAgent (HW3)
- Выполнил zero-shot classification через BART-large-MNLI
- Средняя confidence: 0.899
- Обнаружил 172 примера (13.1%) с confidence < 0.7
- Сгенерировал спецификацию разметки и quality report

### ActiveLearningAgent (HW4)
- Провёл AL-эксперименты с 4 стратегиями: entropy, random, margin, least_confidence
- Начал с 50 labeled → 5 итераций по 20 → 150 labeled
- Обучил финальную модель на полном датасете (TF-IDF + LogisticRegression)

## 3. Human-in-the-Loop

### Точки HITL

1. **Выбор категории и игр** — пользователь выбрал RPG, подтвердил список из 10 игр
2. **Стратегия чистки** — пользователь выбрал стратегии для пропусков, дубликатов, выбросов
3. **Ревью разметки** (главная точка) — проверка 10 low-confidence примеров:
   - Проверено: 10 примеров с confidence 0.00–0.51
   - Исправлено: **6 меток**
   - Типичные ошибки: модель путала ироничные positive-отзывы с negative
   - Паттерны: короткие тексты ("addicting"), смешанные отзывы ("love the game, hate X. 9/10"), сарказм
4. **Выбор стратегий AL** — пользователь добавил margin и least_confidence
5. **Подтверждение модели** — пользователь подтвердил метрики финальной модели

### Влияние HITL
Ручная коррекция 6 из 172 low-confidence меток (3.5%) помогла уточнить граничные случаи. Основной эффект — формирование понимания типов ошибок модели для будущих улучшений.

## 4. Метрики качества

### По этапам

| Этап | Метрика | Значение |
|------|---------|----------|
| Сбор | Всего строк | 1400 |
| Чистка | Строк после чистки | 1311 (-6.4%) |
| Чистка | Дубликатов удалено | 86 |
| Чистка | Пропусков удалено | 3 |
| Разметка | Agreement (auto vs ground truth) | 77.6% |
| Разметка | Cohen's kappa | 0.370 |
| Разметка | Mean confidence | 0.899 |
| Разметка | Low confidence (<0.7) | 172 (13.1%) |
| HITL | Проверено примеров | 10 |
| HITL | Исправлено меток | 6 |

### Active Learning — сравнение стратегий (150 labeled)

| Стратегия | Accuracy | F1 |
|-----------|----------|-----|
| entropy | 0.7148 | 0.6834 |
| margin | 0.7148 | 0.6834 |
| least_confidence | 0.7148 | 0.6834 |
| random | 0.7034 | 0.6669 |

**Вывод**: uncertainty-based стратегии математически эквивалентны при бинарной классификации и все превосходят random sampling на ~1.7% F1.

### Финальная модель (full dataset)

| Метрика | Значение |
|---------|----------|
| **Accuracy** | **0.7529** |
| **F1 (weighted)** | **0.7575** |
| Precision (negative) | 0.59 |
| Recall (negative) | 0.68 |
| Precision (positive) | 0.85 |
| Recall (positive) | 0.79 |

Модель лучше классифицирует positive-отзывы (F1=0.81) из-за дисбаланса классов, несмотря на balanced class weights.

## 5. Ретроспектива

### Что получилось хорошо
- **Полный пайплайн** от сбора до модели работает end-to-end
- **HITL-интеграция** позволила выявить паттерны ошибок (ирония, смешанные отзывы)
- **Active Learning** показал преимущество uncertainty sampling над random
- **Data Card** и annotation spec обеспечивают воспроизводимость

### Что можно улучшить
1. **Дисбаланс классов**: ratio=0.13 — нужны oversampling (SMOTE) или аугментация negative-класса
2. **Zero-shot quality**: κ=0.370 (fair agreement) — стоит использовать fine-tuned модели для разметки
3. **HITL масштаб**: проверено только 10/172 low-confidence — полный ревью повысил бы quality
4. **Модель**: LogisticRegression с TF-IDF — baseline; BERT/transformer модели дадут значительно лучше
5. **Мультиязычность**: только English — Steam имеет отзывы на многих языках
6. **Нейтральный класс**: текущая бинарная разметка теряет нейтральные/смешанные отзывы

### Артефакты проекта

| Файл | Описание |
|------|----------|
| `data/raw/dataset.csv` | Исходные данные (1400 строк) |
| `data/raw/dataset_clean.csv` | Очищенные данные (1311 строк) |
| `data/labeled/dataset_labeled.csv` | Размеченные данные |
| `data/labeled/final_dataset.csv` | Финальный датасет |
| `data/labeled/data_card.md` | Data Card |
| `data/results/al_histories.json` | Истории AL-экспериментов |
| `models/sentiment_model.joblib` | Обученная модель |
| `plots/strategy_comparison.png` | Сравнение стратегий AL |
| `plots/learning_curve.png` | Learning curve |
| `reports/quality_report.md` | Отчёт о качестве данных |
| `reports/annotation_report.md` | Отчёт об авторазметке |
| `reports/al_report.md` | Отчёт об Active Learning |
| `specs/annotation_spec.md` | Спецификация разметки |
| `review_queue.csv` | Low-confidence примеры для ревью |
