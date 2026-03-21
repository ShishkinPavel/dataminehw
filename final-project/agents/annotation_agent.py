"""AnnotationAgent — агент для автоматической разметки, генерации спецификации, оценки качества и экспорта."""

import json
import logging
import os
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score, classification_report
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AnnotationAgent:
    """Агент для автоматической разметки данных.

    Args:
        modality: Модальность данных ('text', 'audio', 'image'). Реализован только 'text'.
        config: Путь к YAML-файлу конфигурации.
    """

    def __init__(self, modality: str = 'text', config: str | None = None) -> None:
        self.modality = modality
        self._config: dict[str, Any] = {}
        if config and os.path.exists(config):
            with open(config, 'r') as f:
                self._config = yaml.safe_load(f) or {}

        self._classifier = None
        auto_label_cfg = self._config.get('auto_label', {})
        self._model_name: str = auto_label_cfg.get('model', 'facebook/bart-large-mnli')
        self._candidate_labels: list[str] = auto_label_cfg.get('candidate_labels', ['positive', 'negative'])
        self._batch_size: int = auto_label_cfg.get('batch_size', 16)
        self._confidence_threshold: float = self._config.get('quality', {}).get('confidence_threshold', 0.7)

        paths = self._config.get('paths', {})
        self._path_labeled: str = paths.get('labeled', 'data/labeled/dataset_labeled.csv')
        self._path_spec: str = paths.get('spec', 'specs/annotation_spec.md')
        self._path_export: str = paths.get('export', 'export/labelstudio_import.json')
        self._path_report: str = paths.get('report', 'reports/quality_report.md')
        self._path_low_confidence: str = paths.get('low_confidence', 'data/low_confidence/flagged_for_review.csv')

    def _get_classifier(self):
        """Lazy-загрузка модели zero-shot classification."""
        if self._classifier is None:
            from transformers import pipeline
            logger.info("Loading model: %s", self._model_name)
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self._model_name,
            )
        return self._classifier

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Автоматическая разметка текстов через zero-shot classification.

        Args:
            df: DataFrame с колонкой 'text'.

        Returns:
            Копия DataFrame с колонками predicted_label и confidence.
        """
        result = df.copy()
        classifier = self._get_classifier()

        texts = result['text'].tolist()
        predicted_labels = []
        confidences = []

        # Batch processing
        for i in tqdm(range(0, len(texts), self._batch_size), desc="Auto-labeling"):
            batch = texts[i:i + self._batch_size]
            clean_batch = []
            batch_indices = []
            for j, text in enumerate(batch):
                if pd.isna(text) or str(text).strip() == '':
                    predicted_labels.append('unknown')
                    confidences.append(0.0)
                else:
                    clean_batch.append(str(text))
                    batch_indices.append(i + j)

            if clean_batch:
                try:
                    results = classifier(clean_batch, self._candidate_labels)
                    if isinstance(results, dict):
                        results = [results]
                    for res in results:
                        predicted_labels.append(res['labels'][0])
                        confidences.append(round(res['scores'][0], 4))
                except Exception:
                    logger.exception("Error in batch %d", i)
                    for _ in clean_batch:
                        predicted_labels.append('unknown')
                        confidences.append(0.0)

        result['predicted_label'] = predicted_labels
        result['confidence'] = confidences

        os.makedirs(os.path.dirname(self._path_labeled), exist_ok=True)
        result.to_csv(self._path_labeled, index=False)
        logger.info("Labeled %d examples, saved to %s", len(result), self._path_labeled)
        return result

    def generate_spec(self, df: pd.DataFrame, task: str = 'sentiment_classification') -> str:
        """Генерирует Markdown-спецификацию разметки.

        Args:
            df: DataFrame с данными.
            task: Название задачи.

        Returns:
            Текст спецификации.
        """
        spec_lines = [
            f"# Спецификация разметки: {task}",
            "",
            "## Задача",
            "Sentiment classification текстовых отзывов. Задача — определить общий эмоциональный тон текста (positive/negative).",
            "",
            "## Классы",
        ]

        for label in self._candidate_labels:
            spec_lines.append(f"\n### {label}")
            if label == 'positive':
                spec_lines.append("**Определение:** Текст выражает положительное отношение, одобрение, удовлетворение.")
            elif label == 'negative':
                spec_lines.append("**Определение:** Текст выражает отрицательное отношение, критику, неудовлетворённость.")
            else:
                spec_lines.append(f"**Определение:** Текст относится к классу '{label}'.")

            spec_lines.append("**Примеры:**")
            label_col = 'label' if 'label' in df.columns else 'predicted_label'
            examples = df[df[label_col] == label].head(3)
            for idx, (_, row) in enumerate(examples.iterrows(), 1):
                text = str(row['text'])[:200]
                if len(str(row['text'])) > 200:
                    text += '...'
                spec_lines.append(f'{idx}. "{text}"')

        spec_lines.extend([
            "",
            "## Граничные случаи",
            '1. **Смешанные отзывы:** "The acting was great but the plot was terrible" — размечать по общему тону',
            '2. **Ирония/сарказм:** "Oh great, another masterpiece..." — negative (несмотря на "great")',
            '3. **Нейтральные:** "The movie was 2 hours long" — если нет явного тона, смотреть контекст',
            "4. **Короткие тексты:** Одно слово/фраза — опираться на коннотацию",
            "",
            "## Инструкция для разметчика",
            "1. Прочитайте весь текст целиком",
            "2. Определите общий тон (positive/negative)",
            '3. При неуверенности — отметьте как "uncertain" в комментариях',
            "4. Время на один пример: ~10-15 секунд",
        ])

        spec_text = "\n".join(spec_lines)

        os.makedirs(os.path.dirname(self._path_spec), exist_ok=True)
        with open(self._path_spec, 'w') as f:
            f.write(spec_text)
        logger.info("Spec saved to %s", self._path_spec)
        return spec_text

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        """Оценка качества авторазметки.

        Args:
            df_labeled: DataFrame с колонками predicted_label и confidence.

        Returns:
            Словарь с метриками качества.
        """
        metrics: dict[str, Any] = {}

        # Label distribution
        metrics['label_dist'] = df_labeled['predicted_label'].value_counts().to_dict()

        # Confidence stats
        conf = df_labeled['confidence']
        metrics['confidence_mean'] = round(conf.mean(), 4)
        metrics['confidence_std'] = round(conf.std(), 4)
        metrics['confidence_below_07'] = int((conf < self._confidence_threshold).sum())

        # Comparison with ground truth
        if 'label' in df_labeled.columns:
            valid = df_labeled.dropna(subset=['label', 'predicted_label'])
            valid = valid[valid['predicted_label'] != 'unknown']
            if len(valid) > 0:
                metrics['kappa'] = round(cohen_kappa_score(valid['label'], valid['predicted_label']), 4)
                metrics['agreement'] = round((valid['label'] == valid['predicted_label']).mean(), 4)
                metrics['classification_report'] = classification_report(
                    valid['label'], valid['predicted_label'], zero_division=0
                )
            else:
                metrics['kappa'] = None
                metrics['agreement'] = None
                metrics['classification_report'] = "No valid examples for comparison."
        else:
            metrics['kappa'] = None
            metrics['agreement'] = None

        # Save report
        self._save_quality_report(metrics, df_labeled)
        return metrics

    def _save_quality_report(self, metrics: dict[str, Any], df_labeled: pd.DataFrame) -> None:
        """Сохраняет отчёт качества в Markdown."""
        lines = [
            "# Quality Report",
            "",
            f"**Total examples:** {len(df_labeled)}",
            f"**Label distribution:** {metrics['label_dist']}",
            "",
            "## Confidence",
            f"- Mean: {metrics['confidence_mean']}",
            f"- Std: {metrics['confidence_std']}",
            f"- Below {self._confidence_threshold}: {metrics['confidence_below_07']}",
            "",
            "## Agreement with Ground Truth",
        ]
        if metrics.get('kappa') is not None:
            lines.append(f"- Cohen's κ: {metrics['kappa']}")
            lines.append(f"- Agreement: {metrics['agreement']}")
            if metrics.get('classification_report'):
                lines.extend(["", "### Classification Report", "```", metrics['classification_report'], "```"])
        else:
            lines.append("Ground truth not available.")

        os.makedirs(os.path.dirname(self._path_report), exist_ok=True)
        with open(self._path_report, 'w') as f:
            f.write("\n".join(lines))
        logger.info("Quality report saved to %s", self._path_report)

    def export_to_labelstudio(
        self, df_labeled: pd.DataFrame, output_path: str | None = None
    ) -> None:
        """Экспорт в формат LabelStudio.

        Args:
            df_labeled: DataFrame с predicted_label и confidence.
            output_path: Путь для сохранения JSON.
        """
        output_path = output_path or self._path_export

        tasks = []
        for _, row in df_labeled.iterrows():
            task = {
                "data": {
                    "text": str(row.get('text', ''))
                },
                "predictions": [
                    {
                        "model_version": "bart-large-mnli-zero-shot",
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "text",
                                "type": "choices",
                                "value": {
                                    "choices": [str(row.get('predicted_label', 'unknown'))]
                                }
                            }
                        ],
                        "score": float(row.get('confidence', 0.0))
                    }
                ]
            }
            tasks.append(task)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        logger.info("Exported %d tasks to %s", len(tasks), output_path)

    def flag_low_confidence(
        self,
        df_labeled: pd.DataFrame,
        threshold: float = 0.7,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """Flag examples with low confidence for human review.

        Args:
            df_labeled: DataFrame with 'confidence' column from auto_label().
            threshold: Confidence threshold. Below this — flagged.
            output_path: Path to save flagged examples.

        Returns:
            DataFrame with flagged examples, sorted by confidence ascending.
        """
        if 'confidence' not in df_labeled.columns:
            raise ValueError("DataFrame must have 'confidence' column. Run auto_label() first.")

        output_path = output_path or self._path_low_confidence

        flagged = df_labeled[df_labeled['confidence'] < threshold].copy()
        flagged = flagged.sort_values('confidence', ascending=True)
        flagged['review_status'] = 'pending'
        flagged['human_label'] = ''
        flagged['reviewer_notes'] = ''

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        flagged.to_csv(output_path, index=False)

        logger.info(
            "Flagged %d/%d examples (%.1f%%) with confidence < %s",
            len(flagged), len(df_labeled),
            len(flagged) / len(df_labeled) * 100 if len(df_labeled) > 0 else 0,
            threshold,
        )

        # Also export in LabelStudio format
        ls_output = output_path.replace('.csv', '_labelstudio.json')
        self.export_to_labelstudio(flagged, output_path=ls_output)

        return flagged
