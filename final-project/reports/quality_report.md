# Data Quality Report

## Dataset Overview
- **Before cleaning**: 1400 rows, 4 columns
- **After cleaning**: 1311 rows, 4 columns
- **Rows removed**: 89

## Issues Found

### Missing Values
- Total: 3
- By column: {'text': 3}
- Strategy: drop

### Duplicates
- Total: 86 (6.14%)
- Strategy: drop (keep first)

### Outliers
- Total: 0 (no numeric columns with outliers)
- Strategy: clip_iqr

### Class Imbalance
- Ratio: 0.1373
- Distribution: {'positive': 1231, 'negative': 169}
- Is imbalanced: True

## After Cleaning
- Missing: 0
- Duplicates: 0
- Imbalance ratio: 0.1302
- Distribution: {'positive': 1160, 'negative': 151}

## Comparison Table

| metric                |    before |     after |   change |
|:----------------------|----------:|----------:|---------:|
| total_rows            | 1400      | 1311      | -89      |
| missing_values        |    3      |    0      |  -3      |
| duplicates            |   86      |    0      | -86      |
| outliers_iqr          |    0      |    0      |   0      |
| label_imbalance_ratio |    0.1373 |    0.1302 |  -0.0071 |
| unique_values_label   |    2      |    2      |   0      |
