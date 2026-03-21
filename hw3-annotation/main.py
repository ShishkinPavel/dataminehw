"""Точка входа. Запуск: python main.py"""
import logging
import pandas as pd
from agents.annotation_agent import AnnotationAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')


def main():
    df = pd.read_csv('data/raw/dataset.csv')
    agent = AnnotationAgent(modality='text', config='config.yaml')

    # Skill 1: Auto-label
    print("=" * 60)
    print("STEP 1: AUTO-LABELING")
    print("=" * 60)
    df_labeled = agent.auto_label(df)
    print(f"Labeled {len(df_labeled)} examples")
    print(f"Label distribution: {df_labeled['predicted_label'].value_counts().to_dict()}")

    # Skill 2: Generate spec
    print("\n" + "=" * 60)
    print("STEP 2: ANNOTATION SPEC")
    print("=" * 60)
    spec = agent.generate_spec(df_labeled, task='sentiment_classification')
    print(f"Spec saved to specs/annotation_spec.md ({len(spec)} chars)")

    # Skill 3: Check quality
    print("\n" + "=" * 60)
    print("STEP 3: QUALITY CHECK")
    print("=" * 60)
    metrics = agent.check_quality(df_labeled)
    for k, v in metrics.items():
        if k != 'classification_report':
            print(f"  {k}: {v}")

    # Skill 4: Export to LabelStudio
    print("\n" + "=" * 60)
    print("STEP 4: EXPORT TO LABEL STUDIO")
    print("=" * 60)
    agent.export_to_labelstudio(df_labeled)
    print("Exported to export/labelstudio_import.json")

    # Bonus: Human-in-the-loop
    print("\n" + "=" * 60)
    print("BONUS: HUMAN-IN-THE-LOOP FLAGGING")
    print("=" * 60)
    flagged = agent.flag_low_confidence(df_labeled, threshold=0.7)
    print(f"Flagged {len(flagged)} examples for human review")
    print(f"Saved to data/low_confidence/flagged_for_review.csv")
    print(f"LabelStudio export: data/low_confidence/flagged_for_review_labelstudio.json")

    print("\nDONE!")


if __name__ == '__main__':
    main()
