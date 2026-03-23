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

    def auto_label(self, df: pd.DataFrame, skip_labeled: bool = False) -> pd.DataFrame:
        """Автоматическая разметка текстов через zero-shot classification.

        Args:
            df: DataFrame с колонкой 'text'.
            skip_labeled: Если True, пропускает строки с уже заполненными
                predicted_label и confidence (ставит их как есть).

        Returns:
            Копия DataFrame с колонками predicted_label, confidence, auto_labeled.
        """
        result = df.copy()

        # Пропуск уже размеченных строк (экономит время при повторном запуске)
        if skip_labeled and 'predicted_label' in result.columns and 'confidence' in result.columns:
            already_done = (
                result['predicted_label'].notna()
                & (result['predicted_label'].astype(str).str.strip() != '')
                & (result['predicted_label'] != 'unknown')
            )
            n_skip = already_done.sum()
            if n_skip > 0:
                logger.info("Skipping %d already-labeled rows, labeling %d new rows", n_skip, len(result) - n_skip)
            if already_done.all():
                result['auto_labeled'] = result['confidence'].astype(float) >= self._confidence_threshold
                os.makedirs(os.path.dirname(self._path_labeled), exist_ok=True)
                result.to_csv(self._path_labeled, index=False)
                return result
            # Only run BART on unlabeled rows; recombine after
            to_label_idx = result.index[~already_done]
            done_idx = result.index[already_done]
        else:
            to_label_idx = result.index
            done_idx = pd.Index([])

        classifier = self._get_classifier()

        # Truncate long texts — BART tokenizer limit is 1024 tokens,
        # but very long reviews degrade classification quality.
        max_chars = self._config.get('auto_label', {}).get('max_text_chars', 512)

        subset = result.loc[to_label_idx]
        texts = subset['text'].tolist()
        predicted_labels = []
        confidences = []

        # Batch processing
        for i in tqdm(range(0, len(texts), self._batch_size), desc="Auto-labeling"):
            batch = texts[i:i + self._batch_size]
            clean_batch = []
            for j, text in enumerate(batch):
                if pd.isna(text) or str(text).strip() == '':
                    predicted_labels.append('unknown')
                    confidences.append(0.0)
                else:
                    truncated = str(text)[:max_chars]
                    clean_batch.append(truncated)

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

        result.loc[to_label_idx, 'predicted_label'] = predicted_labels
        result.loc[to_label_idx, 'confidence'] = confidences

        # Разделение по порогу уверенности: авто-разметка vs ручной ревью
        result['auto_labeled'] = result['confidence'].astype(float) >= self._confidence_threshold
        n_confident = int(result['auto_labeled'].sum())
        n_uncertain = len(result) - n_confident
        logger.info(
            "Labeled %d rows — confident: %d (%.1f%%), uncertain: %d (%.1f%%)",
            len(result), n_confident, n_confident / len(result) * 100,
            n_uncertain, n_uncertain / len(result) * 100,
        )

        os.makedirs(os.path.dirname(self._path_labeled), exist_ok=True)
        result.to_csv(self._path_labeled, index=False)
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
            "",
            "## LabelStudio Config",
            "",
            "```xml",
            "<View>",
            '  <Text name="text" value="$text"/>',
            '  <Choices name="label" toName="text" choice="single" showInline="true">',
        ])
        for label in self._candidate_labels:
            spec_lines.append(f'    <Choice value="{label}"/>')
        spec_lines.extend([
            "  </Choices>",
            "</View>",
            "```",
            "",
            "Вставьте этот XML в LabelStudio → Project Settings → Labeling Interface → Code.",
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
        self,
        df_labeled: pd.DataFrame,
        output_path: str | None = None,
        only_low_confidence: bool = False,
    ) -> None:
        """Экспорт в формат LabelStudio.

        Args:
            df_labeled: DataFrame с predicted_label и confidence.
            output_path: Путь для сохранения JSON.
            only_low_confidence: Если True, экспортирует только строки
                с confidence ниже порога (для ручного ревью).
        """
        output_path = output_path or self._path_export

        df_export = df_labeled
        if only_low_confidence and 'confidence' in df_labeled.columns:
            df_export = df_labeled[df_labeled['confidence'] < self._confidence_threshold]
            logger.info(
                "Filtering to %d low-confidence rows (threshold=%.2f)",
                len(df_export), self._confidence_threshold,
            )

        tasks = []
        for _, row in df_export.iterrows():
            task = {
                "data": {
                    "text": str(row.get('text', '')),
                    "source": str(row.get('source', '')),
                    "gt_label": str(row.get('label', '')),
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

    def generate_plots(self, df_labeled: pd.DataFrame, output_dir: str | None = None) -> str:
        """Визуализация результатов авторазметки.

        Создаёт сводную фигуру из 4 графиков:
        распределение меток, уверенность модели, разброс по классам,
        доля уверенных/неуверенных предсказаний.

        Args:
            df_labeled: DataFrame после auto_label() (predicted_label, confidence).
            output_dir: Куда сохранять. По умолчанию — plots/ рядом с labeled.

        Returns:
            Путь к PNG.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_dir = output_dir or os.path.join(os.path.dirname(self._path_labeled), '..', 'plots')
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'annotation_analysis.png')

        palette = {'positive': '#4C956C', 'negative': '#D64045', 'unknown': '#8B8BAE'}

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # --- 1. Распределение предсказанных меток ---
        vc = df_labeled['predicted_label'].value_counts()
        ax = axes[0, 0]
        bar_colors = [palette.get(l, '#8B8BAE') for l in vc.index]
        bars = ax.bar(vc.index, vc.values, color=bar_colors, edgecolor='#333', linewidth=0.5)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + max(vc.values) * 0.02,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('Количество')
        ax.set_title('Распределение predicted_label (BART)')

        # --- 2. Гистограмма уверенности ---
        ax = axes[0, 1]
        conf_vals = df_labeled['confidence'].astype(float)
        ax.hist(conf_vals, bins=25, color='#5B8FB9', edgecolor='white', linewidth=0.5)
        ax.axvline(self._confidence_threshold, color='#D64045', linestyle='--', linewidth=1.8,
                   label=f'порог = {self._confidence_threshold}')
        below = int((conf_vals < self._confidence_threshold).sum())
        ax.set_title(f'Уверенность модели (ниже порога: {below})')
        ax.set_xlabel('Confidence score')
        ax.set_ylabel('Количество')
        ax.legend(loc='upper left', fontsize=9)

        # --- 3. Boxplot уверенности по классам ---
        ax = axes[1, 0]
        unique_labels = sorted(df_labeled['predicted_label'].dropna().unique())
        groups = [df_labeled.loc[df_labeled['predicted_label'] == l, 'confidence'].astype(float)
                  for l in unique_labels]
        bp = ax.boxplot(groups, labels=unique_labels, patch_artist=True,
                        medianprops={'color': '#333', 'linewidth': 1.5})
        for patch, lbl in zip(bp['boxes'], unique_labels):
            patch.set_facecolor(palette.get(lbl, '#8B8BAE'))
            patch.set_alpha(0.65)
        ax.axhline(self._confidence_threshold, color='#D64045', linestyle=':', linewidth=1.2)
        ax.set_ylabel('Confidence')
        ax.set_title('Уверенность по классам')

        # --- 4. Доля уверенных vs требующих ревью ---
        ax = axes[1, 1]
        if 'auto_labeled' in df_labeled.columns:
            n_ok = int(df_labeled['auto_labeled'].sum())
        else:
            n_ok = int((conf_vals >= self._confidence_threshold).sum())
        n_check = len(df_labeled) - n_ok
        wedges, texts, autotexts = ax.pie(
            [n_ok, n_check],
            labels=[f'Уверенные ({n_ok})', f'На проверку ({n_check})'],
            autopct='%1.1f%%',
            colors=['#4C956C', '#D64045'],
            startangle=140,
            textprops={'fontsize': 10},
        )
        ax.set_title('Авторазметка vs ручной ревью')

        fig.suptitle(f'Анализ авторазметки (n={len(df_labeled)}, model={self._model_name})',
                     fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Annotation plots saved to %s", out_path)
        return out_path

    def get_stats(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        """Собирает метрики аннотации в единый словарь.

        Удобно для программного доступа к результатам (дашборд, пайплайн).

        Args:
            df_labeled: DataFrame после auto_label().

        Returns:
            Словарь с метриками: rows, labels, confidence, threshold, quality.
        """
        conf = df_labeled['confidence'].astype(float)
        n_confident = int((conf >= self._confidence_threshold).sum())

        stats: dict[str, Any] = {
            'rows': len(df_labeled),
            'predicted_labels': df_labeled['predicted_label'].value_counts().to_dict(),
            'confidence': {
                'mean': round(float(conf.mean()), 4),
                'median': round(float(conf.median()), 4),
                'std': round(float(conf.std()), 4),
            },
            'threshold': self._confidence_threshold,
            'n_confident': n_confident,
            'n_uncertain': len(df_labeled) - n_confident,
            'pct_confident': round(n_confident / max(len(df_labeled), 1) * 100, 1),
        }

        # Сравнение с GT, если есть
        if 'label' in df_labeled.columns:
            valid = df_labeled.dropna(subset=['label', 'predicted_label'])
            valid = valid[valid['predicted_label'] != 'unknown']
            if len(valid) > 0:
                disagree_mask = valid['label'] != valid['predicted_label']
                stats['gt_comparison'] = {
                    'kappa': round(float(cohen_kappa_score(valid['label'], valid['predicted_label'])), 4),
                    'agreement': round(float((~disagree_mask).mean()), 4),
                    'disagreements': int(disagree_mask.sum()),
                }

        return stats
