"""Точка входа для сбора данных. Запуск: python main.py"""
import logging
from agents.data_collection_agent import DataCollectionAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)


def main():
    agent = DataCollectionAgent(config='config.yaml')
    df = agent.run()
    print(f"\nCollected {len(df)} rows from {df['source'].nunique()} sources")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    print(f"Saved to {agent.output_path}")


if __name__ == '__main__':
    main()
