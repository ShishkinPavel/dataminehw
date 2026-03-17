"""Точка входа. Запуск: python main.py"""
import logging
import pandas as pd
from agents.data_quality_agent import DataQualityAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')


def main():
    df = pd.read_csv('data/raw/dataset.csv')
    agent = DataQualityAgent(config='config.yaml')

    # Часть 1: Детектив
    print("=" * 60)
    print("PART 1: DETECT ISSUES")
    print("=" * 60)
    report = agent.detect_issues(df)
    print(report['summary'])

    # Часть 2: Хирург — две стратегии
    print("\n" + "=" * 60)
    print("PART 2: FIX — Strategy A (median + clip_iqr)")
    print("=" * 60)
    strategy_a = {'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'}
    df_a = agent.fix(df, strategy=strategy_a)

    print("\n" + "=" * 60)
    print("PART 2: FIX — Strategy B (drop + clip_zscore)")
    print("=" * 60)
    strategy_b = {'missing': 'drop', 'duplicates': 'drop', 'outliers': 'clip_zscore'}
    df_b = agent.fix(df, strategy=strategy_b)

    # Сравнение
    print("\n" + "=" * 60)
    print("COMPARISON: Original vs Strategy A")
    print("=" * 60)
    comp_a = agent.compare(df, df_a)
    print(comp_a.to_string(index=False))

    print("\n" + "=" * 60)
    print("COMPARISON: Original vs Strategy B")
    print("=" * 60)
    comp_b = agent.compare(df, df_b)
    print(comp_b.to_string(index=False))

    # Сохранение лучшей версии (Strategy A как дефолт)
    df_a.to_csv('data/clean/dataset_clean.csv', index=False)

    # Сохранение отчёта
    with open('reports/quality_report.md', 'w') as f:
        f.write('# Data Quality Report\n\n')
        f.write('## Issues Detected\n\n')
        f.write(report['summary'])
        f.write('\n\n## Strategy A: median + clip_iqr\n\n')
        f.write(comp_a.to_markdown(index=False))
        f.write('\n\n## Strategy B: drop + clip_zscore\n\n')
        f.write(comp_b.to_markdown(index=False))

    print(f"\nClean data saved to data/clean/dataset_clean.csv")
    print(f"Report saved to reports/quality_report.md")


if __name__ == '__main__':
    main()
