# AnnotationAgent

Агент для автоматической разметки данных, генерации спецификации, оценки качества и экспорта в LabelStudio.
Домашнее задание №3 по курсу AI Agents.

## Задача ML

Sentiment classification текстовых отзывов. Агент автоматически размечает тексты через zero-shot classification (BART-large-MNLI), оценивает качество разметки и готовит данные для ручной доразметки.

## Архитектура агента

```
AnnotationAgent(modality='text')
│
├── auto_label(df)              → DataFrame + predicted_label, confidence
│   └── facebook/bart-large-mnli (zero-shot classification)
│
├── generate_spec(df, task)     → specs/annotation_spec.md
│   └── задача, классы, примеры, граничные случаи
│
├── check_quality(df_labeled)   → QualityMetrics
│   └── Cohen's κ, agreement %, confidence stats
│
├── export_to_labelstudio(df)   → export/labelstudio_import.json
│   └── формат LabelStudio import с predictions
│
└── flag_low_confidence(df, threshold)  → flagged DataFrame (бонус)
```

## Skills

| Skill | Вход | Выход | Описание |
|-------|------|-------|----------|
| auto_label | DataFrame | DataFrame + predicted_label, confidence | Zero-shot classification |
| generate_spec | DataFrame + task | Markdown-файл | Спецификация для разметчиков |
| check_quality | Labeled DataFrame | dict с метриками | Cohen's κ, distribution, confidence |
| export_to_labelstudio | Labeled DataFrame | JSON | Формат LabelStudio import |
| flag_low_confidence | Labeled DataFrame + threshold | DataFrame | Бонус: HITL фильтрация |

## Быстрый старт

```bash
cd hw3-annotation
pip install -r requirements.txt
python main.py
```

## Использование

```python
from agents.annotation_agent import AnnotationAgent

agent = AnnotationAgent(modality='text')
df_labeled = agent.auto_label(df)
spec = agent.generate_spec(df, task='sentiment_classification')
metrics = agent.check_quality(df_labeled)
agent.export_to_labelstudio(df_labeled)
```

## Воркфлоу с однокурсником

1. Запустить `python main.py` — получить авторазметку
2. Передать `specs/annotation_spec.md` однокурснику
3. Загрузить `export/labelstudio_import.json` в LabelStudio (или дать CSV)
4. Однокурсник размечает 50-100 примеров
5. Сохранить в `data/human/human_labels.csv`
6. Сравнить в ноутбуке

## Формат LabelStudio

JSON-файл готов к импорту: Project → Import → Upload file → labelstudio_import.json

## Метрики качества

Генерируются при запуске `python main.py`, сохраняются в `reports/quality_report.md`.

Ключевые метрики:
- Cohen's κ — согласованность авторазметки с ground truth
- Agreement % — доля совпадений
- Confidence mean/std — уверенность модели
- Below threshold — количество примеров с confidence < 0.7

## Бонус: Human-in-the-loop (+2 балла)

Метод `flag_low_confidence()` автоматически выявляет примеры, где модель неуверена:
- Порог: confidence < 0.7 (настраивается)
- Результат: CSV с колонками review_status, human_label, reviewer_notes
- Дополнительно: LabelStudio JSON для удобного импорта

```python
flagged = agent.flag_low_confidence(df_labeled, threshold=0.7)
# → data/low_confidence/flagged_for_review.csv
# → data/low_confidence/flagged_for_review_labelstudio.json
```

Воркфлоу:
1. Агент размечает все данные автоматически
2. Примеры с confidence < 0.7 → отдельный файл
3. Разметчик проверяет только сложные случаи
4. Экономия: вместо всего датасета нужно проверить только low-confidence примеры

## Структура репозитория

```
hw3-annotation/
├── agents/
│   └── annotation_agent.py
├── config.yaml
├── notebooks/
│   └── annotation_analysis.ipynb
├── data/
│   ├── raw/dataset.csv
│   ├── labeled/dataset_labeled.csv
│   ├── human/
│   └── low_confidence/flagged_for_review.csv
├── specs/
│   └── annotation_spec.md
├── export/
│   └── labelstudio_import.json
├── reports/
│   └── quality_report.md
├── main.py
├── requirements.txt
└── README.md
```

## Требования

Python 3.10+, зависимости: requirements.txt
Первый запуск скачает модель BART (~1.6 GB).
