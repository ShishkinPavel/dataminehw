#!/usr/bin/env python3
"""Create annotation_analysis.ipynb programmatically using nbformat."""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.11.0',
    'mimetype': 'text/x-python',
    'file_extension': '.py',
}

cells = []

# --- 1. Title markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "# Annotation Analysis\n\n"
    "Анализ результатов автоматической разметки текстовых данных.\n"
    "В этом ноутбуке мы исследуем качество авторазметки, распределение классов, "
    "уверенность модели и выявляем проблемные примеры для ручной проверки."
))

# --- 2. Load data ---
cells.append(nbf.v4.new_code_cell("""\
import sys
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from agents.annotation_agent import AnnotationAgent
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

df_raw = pd.read_csv('../data/raw/dataset.csv')
df_labeled = pd.read_csv('../data/labeled/dataset_labeled.csv')
agent = AnnotationAgent(modality='text')

print(f"Shape: {df_labeled.shape}")
print(f"\\nDtypes:\\n{df_labeled.dtypes}")
df_labeled.head()"""))

# --- 3. Distribution markdown ---
cells.append(nbf.v4.new_markdown_cell("## 2. Распределение авторазметки"))

# --- 4. Bar charts ---
cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df_labeled['predicted_label'].value_counts().plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2'))
axes[0].set_title('Распределение predicted_label', fontsize=14)
axes[0].set_ylabel('Количество')

if 'label' in df_labeled.columns:
    comp = pd.DataFrame({
        'original': df_labeled['label'].value_counts(),
        'predicted': df_labeled['predicted_label'].value_counts()
    }).fillna(0)
    comp.plot(kind='bar', ax=axes[1], color=sns.color_palette('Set2'))
    axes[1].set_title('Original vs Predicted labels', fontsize=14)
    axes[1].set_ylabel('Количество')

plt.tight_layout()
plt.show()"""))

# --- 5. Confidence markdown ---
cells.append(nbf.v4.new_markdown_cell("## 3. Анализ уверенности (confidence)"))

# --- 6. Confidence plots ---
cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df_labeled['confidence'], kde=True, ax=axes[0, 0], color=sns.color_palette('Set2')[0])
axes[0, 0].axvline(x=0.7, color='red', linestyle='--', label='threshold=0.7')
axes[0, 0].set_title('Распределение confidence', fontsize=12)
axes[0, 0].legend()

sns.boxplot(data=df_labeled, x='predicted_label', y='confidence', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Confidence по классам', fontsize=12)

if 'text_length' in df_labeled.columns:
    axes[1, 0].scatter(df_labeled['text_length'], df_labeled['confidence'], alpha=0.3, s=10)
    axes[1, 0].set_xlabel('text_length')
    axes[1, 0].set_ylabel('confidence')
    axes[1, 0].set_title('Text length vs Confidence', fontsize=12)

below = (df_labeled['confidence'] < 0.7).sum()
above = (df_labeled['confidence'] >= 0.7).sum()
axes[1, 1].bar(['< 0.7', '>= 0.7'], [below, above], color=['salmon', 'lightgreen'])
axes[1, 1].set_title(f'Примеров ниже порога: {below}', fontsize=12)

plt.tight_layout()
plt.show()"""))

# --- 7. Quality markdown ---
cells.append(nbf.v4.new_markdown_cell("## 4. Качество авторазметки"))

# --- 8. Quality metrics ---
cells.append(nbf.v4.new_code_cell("""\
metrics = agent.check_quality(df_labeled)

print("=== Quality Metrics ===")
for k, v in metrics.items():
    if k != 'classification_report':
        print(f"  {k}: {v}")

if metrics.get('classification_report'):
    print(f"\\n{metrics['classification_report']}")

# Confusion matrix
if 'label' in df_labeled.columns:
    from sklearn.metrics import confusion_matrix
    valid = df_labeled.dropna(subset=['label', 'predicted_label'])
    valid = valid[valid['predicted_label'] != 'unknown']
    labels = sorted(valid['label'].unique())
    cm = confusion_matrix(valid['label'], valid['predicted_label'], labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"Confusion Matrix (κ={metrics.get('kappa', 'N/A')})", fontsize=14)
    plt.tight_layout()
    plt.show()

# Kappa interpretation
kappa = metrics.get('kappa')
if kappa is not None:
    if kappa < 0.2:
        interp = 'poor'
    elif kappa < 0.4:
        interp = 'fair'
    elif kappa < 0.6:
        interp = 'moderate'
    elif kappa < 0.8:
        interp = 'good'
    else:
        interp = 'excellent'
    print(f"\\nCohen's κ = {kappa} → {interp} agreement")"""))

# --- 9. Errors markdown ---
cells.append(nbf.v4.new_markdown_cell("## 5. Примеры ошибок авторазметки"))

# --- 10. Error analysis ---
cells.append(nbf.v4.new_code_cell("""\
if 'label' in df_labeled.columns:
    errors = df_labeled[df_labeled['label'] != df_labeled['predicted_label']].copy()
    print(f"Ошибок: {len(errors)} / {len(df_labeled)} ({len(errors)/len(df_labeled)*100:.1f}%)")

    print("\\nПримеры ошибок:")
    for _, row in errors.head(10).iterrows():
        text_preview = str(row['text'])[:80] + '...' if len(str(row['text'])) > 80 else row['text']
        print(f"  [{row['confidence']:.3f}] pred={row['predicted_label']}, actual={row['label']}: {text_preview}")

    # Confidence distribution: correct vs incorrect
    df_labeled['is_correct'] = df_labeled['label'] == df_labeled['predicted_label']
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df_labeled, x='confidence', hue='is_correct', kde=True, ax=ax, palette=['salmon', 'lightgreen'])
    ax.set_title('Confidence: correct vs incorrect predictions', fontsize=14)
    plt.tight_layout()
    plt.show()
    df_labeled.drop(columns=['is_correct'], inplace=True)"""))

# --- 11. Low confidence markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 6. Примеры с низкой уверенностью\n\n"
    "Эти примеры — кандидаты для ручной разметки."
))

# --- 12. Low confidence examples ---
cells.append(nbf.v4.new_code_cell("""\
low_conf = df_labeled[df_labeled['confidence'] < 0.7].sort_values('confidence')
print(f"Примеров с confidence < 0.7: {len(low_conf)}")

print("\\nТоп-10 с самой низкой уверенностью:")
for _, row in low_conf.head(10).iterrows():
    text_preview = str(row['text'])[:100] + '...' if len(str(row['text'])) > 100 else row['text']
    print(f"  [{row['confidence']:.3f}] {row['predicted_label']}: {text_preview}")"""))

# --- 13. HITL markdown ---
cells.append(nbf.v4.new_markdown_cell("## Бонус: Human-in-the-loop анализ"))

# --- 14. HITL analysis ---
cells.append(nbf.v4.new_code_cell("""\
flagged = agent.flag_low_confidence(df_labeled, threshold=0.7)
print(f"Flagged: {len(flagged)} / {len(df_labeled)} ({len(flagged)/len(df_labeled)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(flagged['confidence'], bins=20, color='salmon', edgecolor='black')
axes[0].set_title('Confidence distribution (flagged examples)')
axes[0].set_xlabel('Confidence')
axes[0].axvline(x=0.7, color='red', linestyle='--', label='threshold=0.7')
axes[0].legend()

flagged['predicted_label'].value_counts().plot(kind='bar', ax=axes[1], color='salmon')
axes[1].set_title('Label distribution (flagged examples)')
axes[1].set_xlabel('Predicted label')
plt.tight_layout()
plt.show()

print("\\nПримеры с самой низкой уверенностью:")
for _, row in flagged.head(5).iterrows():
    text_preview = str(row['text'])[:100] + '...' if len(str(row['text'])) > 100 else row['text']
    print(f"  [{row['confidence']:.3f}] {row['predicted_label']}: {text_preview}")"""))

# --- 15. Spec markdown ---
cells.append(nbf.v4.new_markdown_cell("## 7. Спецификация разметки"))

# --- 16. Display spec ---
cells.append(nbf.v4.new_code_cell("""\
with open('../specs/annotation_spec.md') as f:
    spec_content = f.read()
print(spec_content)"""))

# --- 17. Spec explanation markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "Спецификация обеспечивает согласованность между разметчиками: "
    "определяет классы, примеры и граничные случаи."
))

# --- 18. LabelStudio markdown ---
cells.append(nbf.v4.new_markdown_cell("## 8. Валидация экспорта LabelStudio"))

# --- 19. LabelStudio validation ---
cells.append(nbf.v4.new_code_cell("""\
import json
with open('../export/labelstudio_import.json') as f:
    ls_data = json.load(f)
print(f"Exported {len(ls_data)} tasks")
print(f"Sample task keys: {list(ls_data[0].keys())}")
print(json.dumps(ls_data[0], indent=2, ensure_ascii=False)[:500])"""))

# --- 20. Human comparison markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 9. Сравнение с ручной разметкой\n\n"
    "Для полноценного сравнения:\n"
    "1. Передайте `specs/annotation_spec.md` однокурснику\n"
    "2. Однокурсник размечает 50-100 примеров из `export/labelstudio_import.json`\n"
    "3. Сохраните его разметку в `data/human/human_labels.csv`\n"
    "4. Запустите сравнение ниже"
))

# --- 21. Human comparison code ---
cells.append(nbf.v4.new_code_cell("""\
import os
human_path = '../data/human/human_labels.csv'
if os.path.exists(human_path):
    df_human = pd.read_csv(human_path)
    print(f"Human labels loaded: {len(df_human)}")
else:
    print("Human labels not found. See instructions above.")"""))

# --- 22. Conclusions markdown ---
cells.append(nbf.v4.new_markdown_cell(
    "## 10. Выводы\n\n"
    "- **Качество авторазметки:** Cohen's κ и accuracy генерируются при запуске\n"
    "- **Проблемные зоны:** примеры с confidence < 0.7 — кандидаты для ручной проверки\n"
    "- **Типичные ошибки:** модель путает тон при ирониии и смешанных отзывах\n"
    "- **Рекомендации:** использовать HITL для примеров с низкой уверенностью, "
    "дополнительно обучить модель на domain-specific данных"
))

nb.cells = cells

output_path = os.path.join(os.path.dirname(__file__), 'annotation_analysis.ipynb')
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {output_path}")
