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
            'steam_reviews': self._fetch_steam_reviews,
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

        if 'review_text' in df.columns:
            df = df.rename(columns={'review_text': 'text'})
        if 'review_score' in df.columns:
            df['label'] = df['review_score'].apply(
                lambda x: 'positive' if str(x) == '1' else 'negative'
            )

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

    @staticmethod
    def get_games_by_tag(tag: str, top_n: int = 10) -> list[dict[str, Any]]:
        """Get top indie games by tag from SteamSpy API.

        Args:
            tag: Steam tag (e.g. 'Horror', 'Roguelike', 'Puzzle', 'Platformer').
            top_n: Number of top games to return.

        Returns:
            List of dicts with appid and name.
        """
        logger.info("Fetching top %d indie games for tag '%s' from SteamSpy", top_n, tag)
        try:
            # Fetch games for the requested tag
            resp_tag = requests.get(
                f'https://steamspy.com/api.php?request=tag&tag={tag}',
                timeout=30,
            )
            resp_tag.raise_for_status()
            tag_data = resp_tag.json()

            # Fetch Indie tag to cross-reference (SteamSpy tag endpoint
            # doesn't include per-game tags, so we intersect two lists)
            indie_ids: set[str] = set()
            if tag != 'Indie':
                resp_indie = requests.get(
                    'https://steamspy.com/api.php?request=tag&tag=Indie',
                    timeout=30,
                )
                resp_indie.raise_for_status()
                indie_ids = set(resp_indie.json().keys())
        except (requests.RequestException, ValueError):
            logger.warning("Failed to fetch SteamSpy tag '%s'", tag)
            return []

        # SteamSpy returns dict {appid: {name, ...}}, sorted by owners desc.
        # Steam tags are user-voted, so AAA games often have "Indie" tag.
        # Filter by owners to exclude AAA (>10M owners is not indie).
        MAX_OWNERS = 10_000_000
        games = []
        for appid, info in tag_data.items():
            if tag != 'Indie' and appid not in indie_ids:
                continue
            # Parse owners string like "20,000,000 .. 50,000,000"
            owners_str = info.get('owners', '0 .. 0')
            try:
                upper = int(owners_str.split('..')[-1].strip().replace(',', ''))
            except (ValueError, IndexError):
                upper = 0
            if upper > MAX_OWNERS:
                logger.debug("Skipping '%s' — too many owners (%s)", info.get('name'), owners_str)
                continue
            games.append({'appid': int(appid), 'name': info.get('name', '')})
            if len(games) >= top_n:
                break

        if not games:
            logger.warning("No indie games found for tag '%s'", tag)

        logger.info("Found %d indie games for tag '%s'", len(games), tag)
        return games

    def _fetch_steam_reviews(
        self,
        tag: str = 'Indie',
        top_n: int = 10,
        reviews_per_game: int = 100,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch reviews for top games by Steam tag.

        Args:
            tag: Steam tag to filter games (e.g. 'Horror', 'Puzzle').
            top_n: Number of games to fetch reviews from.
            reviews_per_game: Max reviews per game.

        Returns:
            DataFrame with text, label, source, collected_at.
        """
        games = self.get_games_by_tag(tag, top_n=top_n)
        if not games:
            logger.warning("No games found for tag '%s'", tag)
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        all_reviews: list[dict[str, Any]] = []
        for game in games:
            appid = game['appid']
            name = game['name']
            logger.info("Fetching reviews for '%s' (appid=%d)", name, appid)

            cursor = '*'
            collected = 0
            while collected < reviews_per_game:
                try:
                    resp = requests.get(
                        f'https://store.steampowered.com/appreviews/{appid}',
                        params={
                            'json': '1',
                            'language': 'english',
                            'num_per_page': min(100, reviews_per_game - collected),
                            'cursor': cursor,
                            'filter': 'recent',
                        },
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except (requests.RequestException, ValueError):
                    logger.warning("Failed to fetch reviews for appid %d", appid)
                    break

                reviews = data.get('reviews', [])
                if not reviews:
                    break

                for rev in reviews:
                    all_reviews.append({
                        'text': rev.get('review', ''),
                        'label': 'positive' if rev.get('voted_up') else 'negative',
                        'source': f'steam_{appid}',
                        'collected_at': datetime.now(timezone.utc),
                        'game': name,
                        'tag': tag,
                    })
                    collected += 1

                cursor = data.get('cursor', '')
                if not cursor or collected >= reviews_per_game:
                    break

            logger.info("Collected %d reviews for '%s'", collected, name)

        if not all_reviews:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        df = pd.DataFrame(all_reviews)
        return df[self.REQUIRED_COLUMNS]

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
