---
name: clean
description: "Чистка данных через DataQualityAgent (HW2) — обнаружение и устранение проблем качества"
argument-hint: "[путь к датасету]"
---

# Шаг 2: Чистка данных (DataQualityAgent)

Ты выполняешь этап чистки данных для финального пайплайна.

## Что делать

1. Импортируй `DataQualityAgent` из `hw2-data-quality/agents/data_quality_agent.py`
2. Загрузи данные из `final-project/data/raw/dataset.csv` (или из $ARGUMENTS)
3. Запусти `agent.detect_issues(df)` — покажи отчёт пользователю
4. Примени стратегию чистки через `agent.fix(df, strategy=...)`
5. Сравни до/после через `agent.compare(df_before, df_after)`
6. Сохрани чистые данные в `final-project/data/raw/dataset_clean.csv`
7. Сохрани отчёт в `final-project/reports/quality_report.md`

## Стратегии чистки

- missing: `drop` | `mean` | `median` | `mode` | `ffill`
- duplicates: `drop` | `keep_first` | `keep_last`
- outliers: `clip_iqr` | `clip_zscore` | `drop`

## Агент

Используй `hw2-data-quality/agents/data_quality_agent.py` напрямую.

## HITL — обязательно

После `detect_issues()`:
1. Покажи найденные проблемы (пропуски, дубликаты, выбросы, дисбаланс)
2. Используй `AskUserQuestion` для выбора стратегий:
   - Вопрос 1 (header: "Пропуски"): "Как обработать пропуски?" — drop, mean, median, mode (Recommended), ffill
   - Вопрос 2 (header: "Дубликаты"): "Как обработать дубликаты?" — drop (Recommended), keep_first, keep_last
   - Вопрос 3 (header: "Выбросы"): "Как обработать выбросы?" — clip_iqr (Recommended), clip_zscore, drop
   НЕ задавай вопросы текстом — только через AskUserQuestion.
3. **ОБЯЗАТЕЛЬНО: LLM-рекомендация (бонус HW2, +2 балла).**
   Перед AskUserQuestion вызови `agent.llm_recommend(report, task_description="sentiment classification отзывов на фильмы")`.
   YandexGPT настроен (ключи в `final-project/.env`). Загрузи `.env` через `dotenv.load_dotenv('final-project/.env')` перед вызовом.
   Покажи рекомендацию LLM пользователю рядом с вариантами выбора.
   Если LLM вернёт ошибку — покажи её пользователю, но продолжай работу.
4. После чистки покажи сравнение before/after
5. **Обоснование стратегии (HW2 requirement):** В `quality_report.md` запиши **почему** выбрана каждая стратегия, привязывая к ML-задаче (sentiment analysis). Например: "drop для пропусков в текстовых данных — imputation бессмысленна для текста, пустой отзыв нельзя заполнить средним; clip_iqr для выбросов по длине текста — сохраняет данные лучше чем drop, экстремально длинные/короткие отзывы всё ещё содержат полезный сигнал". Включи в отчёт рекомендацию YandexGPT.

Не применяй стратегию молча — это ключевая HITL-точка.

6. **Если данные оказались чистыми** (0 пропусков, 0 дубликатов, 0 выбросов) — это нормально для курируемых HF-датасетов. В `quality_report.md` добавь абзац:
   > "Данные из курируемых HuggingFace-датасетов ожидаемо чистые. В production-сценарии с пользовательскими отзывами (web-scraping, API социальных сетей) проблем будет значительно больше — пропуски в полях, дубликаты из пагинации, выбросы по длине. Стратегии применены превентивно и готовы к менее чистым данным."
   Это показывает понимание контекста, а не формальность.
