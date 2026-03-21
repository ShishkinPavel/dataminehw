"""Active Learning experiment. Run: python main.py"""

import logging
import json
import os

import pandas as pd

from agents.al_agent import ActiveLearningAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)


def main():
    df_labeled = pd.read_csv('data/splits/labeled.csv')
    df_pool = pd.read_csv('data/splits/pool.csv')
    df_test = pd.read_csv('data/splits/test.csv')

    agent = ActiveLearningAgent(model='logreg', config='config.yaml')
    histories = {}

    # Run entropy strategy
    print("=" * 60)
    print("STRATEGY: entropy")
    print("=" * 60)
    hist_entropy = agent.run_cycle(
        labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
        strategy='entropy', n_iterations=5, batch_size=20
    )
    histories['entropy'] = hist_entropy

    # Run margin strategy
    print("\n" + "=" * 60)
    print("STRATEGY: margin")
    print("=" * 60)
    hist_margin = agent.run_cycle(
        labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
        strategy='margin', n_iterations=5, batch_size=20
    )
    histories['margin'] = hist_margin

    # Run random baseline
    print("\n" + "=" * 60)
    print("STRATEGY: random")
    print("=" * 60)
    hist_random = agent.run_cycle(
        labeled_df=df_labeled, pool_df=df_pool, test_df=df_test,
        strategy='random', n_iterations=5, batch_size=20
    )
    histories['random'] = hist_random

    # Compare
    ActiveLearningAgent.compare_strategies(histories)
    print("\nStrategy comparison saved to plots/strategy_comparison.png")

    # Save results
    os.makedirs('data/results', exist_ok=True)
    with open('data/results/histories.json', 'w') as f:
        json.dump(histories, f, indent=2)

    # Report
    os.makedirs('reports', exist_ok=True)
    final_rnd = hist_random[-1]

    target_acc = final_rnd['accuracy']
    saved = 0
    for rec in hist_entropy:
        if rec['accuracy'] >= target_acc:
            saved = final_rnd['n_labeled'] - rec['n_labeled']
            break

    with open('reports/al_report.md', 'w') as f:
        f.write('# Active Learning Report\n\n')
        f.write('## Strategies Compared\n\n')
        f.write('| Strategy | Final Accuracy | Final F1 | N Labeled |\n')
        f.write('|----------|---------------|----------|----------|\n')
        for name, hist in histories.items():
            final = hist[-1]
            f.write(f"| {name} | {final['accuracy']:.4f} | {final['f1']:.4f} | {final['n_labeled']} |\n")
        f.write(f'\n## Savings\n\n')
        f.write(f'Entropy strategy reached random baseline accuracy ({target_acc:.4f}) ')
        f.write(f'with **{saved} fewer labeled examples**.\n')

    print(f"\nReport saved to reports/al_report.md")
    print(f"Entropy saved ~{saved} examples vs random baseline")

    # Bonus: LLM analysis
    print("\n" + "=" * 60)
    print("BONUS: LLM ANALYSIS")
    print("=" * 60)
    recommendation = agent.llm_recommend_strategy(histories)
    print(recommendation)


if __name__ == '__main__':
    main()
