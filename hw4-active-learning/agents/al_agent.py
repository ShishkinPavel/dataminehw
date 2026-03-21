"""Active Learning Agent for smart data selection."""

import logging
import os
import json

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class ActiveLearningAgent:
    """Agent for Active Learning experiments.

    Trains a baseline model, selects the most informative examples
    from a pool, retrains, and compares selection strategies.

    Args:
        model: Model type - 'logreg', 'svm', or 'rf'.
        config: Optional path to YAML config file.
    """

    def __init__(self, model: str = 'logreg', config: str | None = None) -> None:
        self.model_name = model
        self.pipeline = None
        self.config: dict = {}
        if config and os.path.exists(config):
            import yaml
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f) or {}

    def _get_classifier(self):
        """Create classifier instance based on model_name."""
        if self.model_name == 'logreg':
            return LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        elif self.model_name == 'svm':
            return SVC(probability=True, random_state=42)
        elif self.model_name == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model: {self.model_name}. Use 'logreg', 'svm', or 'rf'.")

    def fit(self, labeled_df: pd.DataFrame) -> None:
        """Train the model on labeled data.

        Args:
            labeled_df: DataFrame with 'text' and 'label' columns.
        """
        try:
            tfidf_cfg = self.config.get('tfidf', {})
            max_features = tfidf_cfg.get('max_features', 5000)
            ngram_range = tuple(tfidf_cfg.get('ngram_range', [1, 2]))

            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
                ('clf', self._get_classifier())
            ])
            self.pipeline.fit(labeled_df['text'], labeled_df['label'])
            logger.info(f"Fitted on {len(labeled_df)} examples, classes: {labeled_df['label'].nunique()}")
        except Exception as e:
            logger.error(f"Error during fit: {e}")
            raise

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: str = 'entropy',
        batch_size: int = 20,
        iteration: int = 0,
    ) -> list[int]:
        """Select the most informative examples from the pool.

        Args:
            pool_df: DataFrame with 'text' column (unlabeled pool).
            strategy: Selection strategy - 'entropy', 'least_confidence',
                      'margin', or 'random'.
            batch_size: Number of examples to select.
            iteration: Current AL iteration (used as seed offset for random).

        Returns:
            List of indices (positions in pool_df).
        """
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        batch_size = min(batch_size, len(pool_df))

        if strategy == 'entropy':
            proba = self.pipeline.predict_proba(pool_df['text'])
            entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            indices = np.argsort(entropy)[-batch_size:].tolist()
        elif strategy == 'least_confidence':
            proba = self.pipeline.predict_proba(pool_df['text'])
            max_proba = np.max(proba, axis=1)
            indices = np.argsort(max_proba)[:batch_size].tolist()
        elif strategy == 'margin':
            proba = self.pipeline.predict_proba(pool_df['text'])
            sorted_proba = np.sort(proba, axis=1)
            margin = sorted_proba[:, -1] - sorted_proba[:, -2]
            indices = np.argsort(margin)[:batch_size].tolist()
        elif strategy == 'random':
            indices = np.random.RandomState(42 + iteration).choice(
                len(pool_df), size=batch_size, replace=False
            ).tolist()
        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                "Use 'entropy', 'least_confidence', 'margin', or 'random'."
            )

        logger.info(f"Queried {batch_size} examples using '{strategy}' strategy")
        return indices

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate the model on test data.

        Args:
            test_df: DataFrame with 'text' and 'label' columns.

        Returns:
            Dict with 'accuracy', 'f1', and 'predictions'.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        try:
            preds = self.pipeline.predict(test_df['text'])
            return {
                'accuracy': accuracy_score(test_df['label'], preds),
                'f1': f1_score(test_df['label'], preds, average='weighted'),
                'predictions': preds
            }
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: str = 'entropy',
        n_iterations: int = 5,
        batch_size: int = 20
    ) -> list[dict]:
        """Run a full Active Learning cycle.

        Args:
            labeled_df: Initial labeled data.
            pool_df: Unlabeled pool data.
            test_df: Test data for evaluation.
            strategy: Selection strategy.
            n_iterations: Number of AL iterations.
            batch_size: Examples to add per iteration.

        Returns:
            List of dicts with iteration metrics.
        """
        history = []
        current_labeled = labeled_df.copy()
        current_pool = pool_df.copy()

        for i in range(n_iterations + 1):
            self.fit(current_labeled)
            metrics = self.evaluate(test_df)

            record = {
                'iteration': i,
                'n_labeled': len(current_labeled),
                'accuracy': round(metrics['accuracy'], 4),
                'f1': round(metrics['f1'], 4),
                'strategy': strategy
            }
            history.append(record)
            logger.info(
                f"Iter {i}: n={len(current_labeled)}, "
                f"acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}"
            )

            if i < n_iterations and len(current_pool) >= batch_size:
                indices = self.query(current_pool, strategy=strategy, batch_size=batch_size, iteration=i)
                selected = current_pool.iloc[indices]
                current_labeled = pd.concat([current_labeled, selected], ignore_index=True)
                current_pool = current_pool.drop(current_pool.index[indices]).reset_index(drop=True)

        return history

    def report(self, history: list[dict], label: str = '', output_dir: str = 'plots') -> None:
        """Generate learning curve plot.

        Args:
            history: List of iteration metrics from run_cycle.
            label: Label for the legend.
            output_dir: Directory to save the plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        df_hist = pd.DataFrame(history)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(
            df_hist['n_labeled'], df_hist['accuracy'], 'o-',
            label=label or df_hist['strategy'].iloc[0]
        )
        axes[0].set_xlabel('Number of labeled examples')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Learning Curve — Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(
            df_hist['n_labeled'], df_hist['f1'], 's-',
            label=label or df_hist['strategy'].iloc[0]
        )
        axes[1].set_xlabel('Number of labeled examples')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Learning Curve — F1')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/learning_curve.png', dpi=150)
        plt.close()
        logger.info(f"Learning curve saved to {output_dir}/learning_curve.png")

    @staticmethod
    def compare_strategies(histories: dict[str, list[dict]], output_dir: str = 'plots') -> None:
        """Plot multiple strategies on one chart.

        Args:
            histories: Dict mapping strategy name to history list.
            output_dir: Directory to save the plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = {'entropy': 'tab:blue', 'least_confidence': 'tab:green', 'margin': 'tab:orange', 'random': 'tab:gray'}

        for name, history in histories.items():
            df_h = pd.DataFrame(history)
            c = colors.get(name, None)
            axes[0].plot(df_h['n_labeled'], df_h['accuracy'], 'o-', label=name, color=c)
            axes[1].plot(df_h['n_labeled'], df_h['f1'], 's-', label=name, color=c)

        for ax, metric in zip(axes, ['Accuracy', 'F1']):
            ax.set_xlabel('Number of labeled examples')
            ax.set_ylabel(metric)
            ax.set_title(f'Active Learning — {metric}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/strategy_comparison.png', dpi=150)
        plt.close()
        logger.info(f"Strategy comparison saved to {output_dir}/strategy_comparison.png")

    def _call_yandexgpt(self, prompt: str, max_tokens: int = 512) -> str:
        """Call YandexGPT API.

        Args:
            prompt: User prompt text.
            max_tokens: Maximum tokens in response.

        Returns:
            Response text or error message.
        """
        import requests

        api_key = os.environ.get('YANDEX_API_KEY')
        folder_id = os.environ.get('YANDEX_FOLDER_ID')
        if not api_key or not folder_id:
            return "YANDEX_API_KEY or YANDEX_FOLDER_ID not set"

        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        body = {
            "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False,
                "temperature": 0.3,
                "maxTokens": max_tokens,
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
            resp = requests.post(url, json=body, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["result"]["alternatives"][0]["message"]["text"]
        except Exception as e:
            return f"LLM error: {e}"

    def llm_explain_selection(
        self,
        selected_texts: list[str],
        strategy: str,
        iteration: int
    ) -> str:
        """Ask YandexGPT to explain why selected examples are informative.

        Args:
            selected_texts: Texts chosen by the query strategy.
            strategy: Strategy name ('entropy', 'margin', 'random').
            iteration: Current AL iteration number.

        Returns:
            String with LLM explanation.
        """
        examples = "\n".join(
            f"{i+1}. {t[:200]}..." if len(t) > 200 else f"{i+1}. {t}"
            for i, t in enumerate(selected_texts[:5])
        )

        prompt = f"""You are an Active Learning expert. Analyze these examples selected by the '{strategy}' strategy at iteration {iteration}.

Selected examples:
{examples}

Questions:
1. What makes these examples potentially informative for a sentiment classifier?
2. Do you see patterns (ambiguous sentiment, mixed opinions, sarcasm, unusual vocabulary)?
3. Would a human annotator find these examples easy or hard to label?

Respond in Russian, 3-5 sentences total. Be specific about the examples."""

        return self._call_yandexgpt(prompt)

    def llm_recommend_strategy(self, histories: dict[str, list[dict]]) -> str:
        """Ask YandexGPT to recommend the best strategy based on results.

        Args:
            histories: Dict mapping strategy name to history list.

        Returns:
            String with recommendation.
        """
        summary = json.dumps(histories, indent=2)

        prompt = f"""You are an Active Learning expert. Compare these AL strategies based on experiment results:

{summary}

Analyze:
1. Which strategy is most efficient (best quality with fewest labels)?
2. How many labeled examples does the best strategy save vs random?
3. Practical recommendation: which strategy and why?

Respond in Russian, structured and concise."""

        return self._call_yandexgpt(prompt)
