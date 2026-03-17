"""DataCollectionAgent — агент для сбора данных из нескольких источников."""

import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DataCollectionAgent:
    """Агент для сбора и унификации данных из нескольких источников.

    Args:
        config: Путь к YAML-файлу конфигурации.
    """

    REQUIRED_COLUMNS = ['text', 'label', 'source', 'collected_at']

    def __init__(self, config: str = 'config.yaml') -> None:
        with open(config, 'r') as f:
            self._config = yaml.safe_load(f)

        self.output_path: str = self._config.get('output', {}).get('path', 'data/raw/dataset.csv')
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def run(self, sources: list[dict[str, Any]] | None = None) -> pd.DataFrame:
        """Запускает сбор данных из всех источников.

        Args:
            sources: Список источников. Если None — берётся из config.yaml.

        Returns:
            DataFrame с колонками: text, label, source, collected_at.
        """
        if sources is None:
            sources = self._config.get('sources', [])

        dfs: list[pd.DataFrame] = []
        skill_map = {
            'hf_dataset': self._load_dataset,
            'scrape': self._scrape,
            'api': self._fetch_api,
        }

        for src in sources:
            src_type = src.get('type', '')
            skill = skill_map.get(src_type)
            if skill is None:
                logger.warning("Unknown source type: %s", src_type)
                continue

            params = {k: v for k, v in src.items() if k != 'type'}
            try:
                df = skill(**params)
                logger.info("Source '%s': collected %d rows", src_type, len(df))
                dfs.append(df)
            except Exception:
                logger.exception("Error collecting from source '%s'", src_type)
                dfs.append(pd.DataFrame(columns=self.REQUIRED_COLUMNS))

        result = self._merge(dfs)
        result.to_csv(self.output_path, index=False)
        logger.info("Saved %d rows to %s", len(result), self.output_path)
        return result

    def _load_dataset(
        self,
        name: str,
        split: str = 'train',
        sample_size: int = 1000,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Загружает датасет из HuggingFace.

        Args:
            name: Имя датасета на HuggingFace.
            split: Сплит датасета.
            sample_size: Размер выборки.

        Returns:
            DataFrame с унифицированной схемой.
        """
        from datasets import load_dataset

        logger.info("Loading HuggingFace dataset '%s' (split=%s, sample=%d)", name, split, sample_size)
        dataset = load_dataset(name, split=split)
        dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
        df = dataset.to_pandas()

        # Маппинг для imdb
        if name == 'imdb':
            label_map = {0: 'negative', 1: 'positive'}
            df['label'] = df['label'].map(label_map)

        df['source'] = f'hf_{name}'
        df['collected_at'] = datetime.now(timezone.utc)
        return df[self.REQUIRED_COLUMNS]

    def _scrape(
        self,
        url: str,
        selector: str,
        label: str = 'unknown',
        source_name: str = 'scrape',
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Собирает данные через web scraping.

        Args:
            url: URL для скрапинга.
            selector: CSS-селектор для извлечения текста.
            label: Метка класса для собранных данных.
            source_name: Идентификатор источника.

        Returns:
            DataFrame с унифицированной схемой.
        """
        logger.info("Scraping %s (selector='%s')", url, selector)
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            logger.warning("Failed to fetch %s", url)
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        soup = BeautifulSoup(response.text, 'html.parser')
        elements = soup.select(selector)
        texts = [el.get_text(strip=True) for el in elements]

        if not texts:
            logger.warning("No elements found for selector '%s' at %s", selector, url)
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        return pd.DataFrame({
            'text': texts,
            'label': label,
            'source': source_name,
            'collected_at': datetime.now(timezone.utc),
        })

    def _fetch_api(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        text_field: str = 'text',
        label_field: str | None = None,
        source_name: str = 'api',
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Получает данные через REST API.

        Args:
            endpoint: URL эндпоинта.
            params: Query-параметры запроса.
            text_field: Имя поля с текстом в ответе API.
            label_field: Имя поля с меткой в ответе API.
            source_name: Идентификатор источника.

        Returns:
            DataFrame с унифицированной схемой.
        """
        logger.info("Fetching API: %s", endpoint)
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            logger.warning("Failed to fetch API %s", endpoint)
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        if isinstance(data, dict):
            data = data.get('results', data.get('data', []))

        df = pd.DataFrame(data)
        if text_field in df.columns:
            df = df.rename(columns={text_field: 'text'})
        if label_field and label_field in df.columns:
            df = df.rename(columns={label_field: 'label'})
        elif 'label' not in df.columns:
            df['label'] = 'unknown'

        df['source'] = source_name
        df['collected_at'] = datetime.now(timezone.utc)

        return df[self.REQUIRED_COLUMNS] if 'text' in df.columns else pd.DataFrame(columns=self.REQUIRED_COLUMNS)

    def _merge(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Объединяет DataFrames из разных источников.

        Args:
            dfs: Список DataFrames для объединения.

        Returns:
            Объединённый DataFrame с гарантированной схемой.
        """
        if not dfs:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        result = pd.concat(dfs, ignore_index=True)

        for col in self.REQUIRED_COLUMNS:
            if col not in result.columns:
                result[col] = 'unknown' if col != 'collected_at' else datetime.now(timezone.utc)

        return result
