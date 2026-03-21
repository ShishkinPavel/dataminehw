"""DataQualityAgent — агент-детектив для обнаружения и устранения проблем качества данных."""

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityAgent:
    """Агент для автоматического обнаружения и устранения проблем качества данных.

    Args:
        config: Путь к YAML-файлу конфигурации (опционально).
    """

    def __init__(self, config: str | None = None) -> None:
        self._config: dict[str, Any] = {}
        if config and os.path.exists(config):
            with open(config, 'r') as f:
                self._config = yaml.safe_load(f) or {}

        self.target_column: str = self._config.get('target_column', 'label')

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        """Обнаруживает проблемы качества данных.

        Args:
            df: Исходный DataFrame для анализа.

        Returns:
            Словарь QualityReport с ключами: missing, duplicates, outliers, imbalance, summary.
        """
        report: dict[str, Any] = {}

        try:
            report['missing'] = self._detect_missing(df)
        except Exception:
            logger.exception("Error detecting missing values")
            report['missing'] = {'total': 0, 'by_column': {}, 'percent_by_column': {}}

        try:
            report['duplicates'] = self._detect_duplicates(df)
        except Exception:
            logger.exception("Error detecting duplicates")
            report['duplicates'] = {'total': 0, 'percent': 0.0, 'examples': pd.DataFrame()}

        try:
            report['outliers'] = self._detect_outliers(df)
        except Exception:
            logger.exception("Error detecting outliers")
            report['outliers'] = {'by_column': {}}

        try:
            report['imbalance'] = self._detect_imbalance(df)
        except Exception:
            logger.exception("Error detecting imbalance")
            report['imbalance'] = {
                'target_column': self.target_column,
                'distribution': {},
                'ratio': 1.0,
                'is_imbalanced': False,
            }

        report['summary'] = self._build_summary(report, df)
        return report

    def fix(self, df: pd.DataFrame, strategy: dict[str, str] | None = None) -> pd.DataFrame:
        """Исправляет проблемы качества данных.

        Args:
            df: Исходный DataFrame.
            strategy: Словарь стратегий: {'missing': ..., 'duplicates': ..., 'outliers': ...}.

        Returns:
            Очищенная копия DataFrame.
        """
        if strategy is None:
            strategy = self._config.get('default_strategy', {
                'missing': 'median',
                'duplicates': 'drop',
                'outliers': 'clip_iqr',
            })

        result = df.copy()

        # Порядок: duplicates → missing → outliers
        try:
            result = self._fix_duplicates(result, strategy.get('duplicates', 'drop'))
        except Exception:
            logger.exception("Error fixing duplicates")

        try:
            result = self._fix_missing(result, strategy.get('missing', 'median'))
        except Exception:
            logger.exception("Error fixing missing values")

        try:
            result = self._fix_outliers(result, strategy.get('outliers', 'clip_iqr'))
        except Exception:
            logger.exception("Error fixing outliers")

        return result.reset_index(drop=True)

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """Сравнивает два DataFrame по ключевым метрикам качества.

        Args:
            df_before: Исходный DataFrame.
            df_after: Очищенный DataFrame.

        Returns:
            DataFrame с колонками: metric, before, after, change.
        """
        report_before = self.detect_issues(df_before)
        report_after = self.detect_issues(df_after)

        numeric_cols = df_before.select_dtypes(include='number').columns.tolist()

        metrics = [
            ('total_rows', len(df_before), len(df_after)),
            ('missing_values', report_before['missing']['total'], report_after['missing']['total']),
            ('duplicates', report_before['duplicates']['total'], report_after['duplicates']['total']),
        ]

        # Outliers по IQR
        outliers_before = sum(
            info.get('iqr', {}).get('count', 0)
            for info in report_before['outliers']['by_column'].values()
        )
        outliers_after = sum(
            info.get('iqr', {}).get('count', 0)
            for info in report_after['outliers']['by_column'].values()
        )
        metrics.append(('outliers_iqr', outliers_before, outliers_after))

        # Средние значения числовых колонок
        for col in numeric_cols:
            if col in df_after.columns:
                mean_before = round(df_before[col].mean(), 2) if not df_before[col].isna().all() else 0
                mean_after = round(df_after[col].mean(), 2) if not df_after[col].isna().all() else 0
                metrics.append((f'mean_{col}', mean_before, mean_after))

        # Imbalance ratio
        metrics.append((
            'label_imbalance_ratio',
            round(report_before['imbalance']['ratio'], 4),
            round(report_after['imbalance']['ratio'], 4),
        ))

        # Unique labels
        if self.target_column in df_before.columns and self.target_column in df_after.columns:
            metrics.append((
                f'unique_values_{self.target_column}',
                df_before[self.target_column].nunique(),
                df_after[self.target_column].nunique(),
            ))

        rows = []
        for name, before, after in metrics:
            change = round(after - before, 4) if isinstance(after, (int, float)) and isinstance(before, (int, float)) else None
            rows.append({'metric': name, 'before': before, 'after': after, 'change': change})

        return pd.DataFrame(rows)

    def llm_recommend(self, report: dict, task_description: str = "") -> str:
        """LLM-рекомендация через YandexGPT API.

        Args:
            report: Output of detect_issues().
            task_description: Description of the ML task.

        Returns:
            String with LLM recommendation.

        Env vars:
            YANDEX_API_KEY: API-ключ Yandex Cloud.
            YANDEX_FOLDER_ID: ID каталога в Yandex Cloud.
        """
        import requests as req

        api_key = os.environ.get('YANDEX_API_KEY')
        folder_id = os.environ.get('YANDEX_FOLDER_ID')
        if not api_key:
            return "YANDEX_API_KEY not set. Export it: export YANDEX_API_KEY=..."
        if not folder_id:
            return "YANDEX_FOLDER_ID not set. Export it: export YANDEX_FOLDER_ID=..."

        report_summary = {
            'missing': {
                'total': report['missing']['total'],
                'by_column': report['missing']['by_column'],
                'percent_by_column': report['missing']['percent_by_column'],
            },
            'duplicates': {
                'total': report['duplicates']['total'],
                'percent': report['duplicates']['percent'],
            },
            'outliers': {
                col: {
                    method: {'count': info['count']}
                    for method, info in methods.items()
                }
                for col, methods in report['outliers']['by_column'].items()
            },
            'imbalance': {
                'ratio': report['imbalance']['ratio'],
                'is_imbalanced': report['imbalance']['is_imbalanced'],
                'distribution': report['imbalance']['distribution'],
            },
        }

        prompt = f"""Ты — эксперт по качеству данных. Проанализируй отчёт и рекомендуй лучшую стратегию очистки.

ML-задача: {task_description or 'не указана'}

Отчёт о качестве:
{json.dumps(report_summary, indent=2, default=str, ensure_ascii=False)}

Доступные стратегии:
- missing: mean, median, mode, drop, ffill
- duplicates: drop, keep_first, keep_last
- outliers: clip_iqr, clip_zscore, drop

Ответь по структуре:
1. Краткий анализ проблем (2-3 предложения)
2. Рекомендуемая стратегия (конкретный dict для agent.fix())
3. Обоснование выбора (почему именно эта стратегия для данной ML-задачи)
"""

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        body = {
            "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,
                "maxTokens": 1024,
            },
            "messages": [
                {"role": "user", "text": prompt},
            ],
        }
        headers = {
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = req.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            return result["result"]["alternatives"][0]["message"]["text"]
        except Exception as e:
            logger.exception("LLM API error")
            return f"LLM API error: {e}"

    # ── Private: Detection ────────────────────────────────────────

    def _detect_missing(self, df: pd.DataFrame) -> dict[str, Any]:
        """Обнаруживает пропущенные значения."""
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        total = int(missing_counts.sum())
        by_column = missing_counts.to_dict()
        percent_by_column = {col: round(cnt / len(df) * 100, 2) for col, cnt in by_column.items()}
        logger.info("Missing values: %d total in %d columns", total, len(by_column))
        return {'total': total, 'by_column': by_column, 'percent_by_column': percent_by_column}

    def _detect_duplicates(self, df: pd.DataFrame) -> dict[str, Any]:
        """Обнаруживает дубликаты."""
        dup_mask = df.duplicated(keep='first')
        total = int(dup_mask.sum())
        percent = round(total / len(df) * 100, 2)
        examples = df[dup_mask].head(5)
        logger.info("Duplicates: %d (%.2f%%)", total, percent)
        return {'total': total, 'percent': percent, 'examples': examples}

    def _detect_outliers(self, df: pd.DataFrame) -> dict[str, Any]:
        """Обнаруживает выбросы в числовых колонках (IQR и z-score)."""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        by_column: dict[str, Any] = {}

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) == 0:
                continue

            # IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_count = int(((series < lower) | (series > upper)).sum())

            # Z-score
            z_scores = np.abs(stats.zscore(series))
            zscore_count = int((z_scores > 3).sum())

            by_column[col] = {
                'iqr': {'count': iqr_count, 'bounds': (round(lower, 2), round(upper, 2))},
                'zscore': {'count': zscore_count, 'threshold': 3.0},
            }
            logger.info("Outliers in '%s': IQR=%d, Z-score=%d", col, iqr_count, zscore_count)

        return {'by_column': by_column}

    def _detect_imbalance(self, df: pd.DataFrame) -> dict[str, Any]:
        """Обнаруживает дисбаланс классов."""
        if self.target_column not in df.columns:
            return {
                'target_column': self.target_column,
                'distribution': {},
                'ratio': 1.0,
                'is_imbalanced': False,
            }

        dist = df[self.target_column].value_counts().to_dict()
        counts = list(dist.values())
        ratio = round(min(counts) / max(counts), 4) if counts else 1.0
        is_imbalanced = ratio < 0.5
        logger.info("Imbalance: ratio=%.4f, imbalanced=%s", ratio, is_imbalanced)
        return {
            'target_column': self.target_column,
            'distribution': dist,
            'ratio': ratio,
            'is_imbalanced': is_imbalanced,
        }

    def _build_summary(self, report: dict[str, Any], df: pd.DataFrame) -> str:
        """Строит человекочитаемое резюме."""
        lines = ["=== Data Quality Report ===", f"Dataset: {len(df)} rows, {len(df.columns)} columns", ""]

        m = report['missing']
        lines.append(f"Missing values: {m['total']} total in {len(m['by_column'])} column(s)")
        for col, cnt in m['by_column'].items():
            lines.append(f"  - {col}: {cnt} ({m['percent_by_column'][col]}%)")

        d = report['duplicates']
        lines.append(f"\nDuplicates: {d['total']} ({d['percent']}%)")

        o = report['outliers']
        total_outliers = sum(v['iqr']['count'] for v in o['by_column'].values())
        lines.append(f"\nOutliers (IQR): {total_outliers} total in {len(o['by_column'])} column(s)")
        for col, info in o['by_column'].items():
            lines.append(f"  - {col}: IQR={info['iqr']['count']}, Z-score={info['zscore']['count']}, bounds={info['iqr']['bounds']}")

        i = report['imbalance']
        lines.append(f"\nClass imbalance ({i['target_column']}): ratio={i['ratio']}, imbalanced={i['is_imbalanced']}")
        for label, count in i['distribution'].items():
            lines.append(f"  - {label}: {count}")

        return "\n".join(lines)

    # ── Private: Fixing ───────────────────────────────────────────

    def _fix_missing(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Заполняет или удаляет пропуски."""
        logger.info("Fixing missing values: strategy='%s'", strategy)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        if strategy == 'drop':
            return df.dropna()
        elif strategy == 'ffill':
            return df.ffill()
        elif strategy in ('mean', 'median', 'mode'):
            for col in numeric_cols:
                if df[col].isnull().any():
                    if strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        mode_val = df[col].mode()
                        df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)
            for col in non_numeric_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'unknown')
            return df

        logger.warning("Unknown missing strategy: '%s', skipping", strategy)
        return df

    def _fix_duplicates(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Удаляет дубликаты."""
        logger.info("Fixing duplicates: strategy='%s'", strategy)
        if strategy in ('drop', 'keep_first'):
            return df.drop_duplicates(keep='first')
        elif strategy == 'keep_last':
            return df.drop_duplicates(keep='last')

        logger.warning("Unknown duplicates strategy: '%s', skipping", strategy)
        return df

    def _fix_outliers(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Обрабатывает выбросы в числовых колонках."""
        logger.info("Fixing outliers: strategy='%s'", strategy)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if strategy == 'clip_iqr':
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)
        elif strategy == 'clip_zscore':
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std
                df[col] = df[col].clip(lower, upper)
        elif strategy == 'drop':
            mask = pd.Series(True, index=df.index)
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask &= (df[col] >= lower) & (df[col] <= upper) | df[col].isna()
            df = df[mask]
        else:
            logger.warning("Unknown outliers strategy: '%s', skipping", strategy)

        return df
