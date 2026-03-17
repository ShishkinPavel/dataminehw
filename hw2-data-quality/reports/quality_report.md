# Data Quality Report

## Issues Detected

=== Data Quality Report ===
Dataset: 1040 rows, 6 columns

Missing values: 80 total in 2 column(s)
  - text_length: 29 (2.79%)
  - rating: 51 (4.9%)

Duplicates: 30 (2.88%)

Outliers (IQR): 97 total in 2 column(s)
  - text_length: IQR=97, Z-score=15, bounds=(np.float64(-601.5), np.float64(2886.5))
  - rating: IQR=0, Z-score=0, bounds=(np.float64(-1.14), np.float64(7.02))

Class imbalance (label): ratio=0.0189, imbalanced=True
  - negative: 530
  - positive: 500
  - quote: 10

## Strategy A: median + clip_iqr

| metric                |    before |     after |    change |
|:----------------------|----------:|----------:|----------:|
| total_rows            | 1040      | 1010      |  -30      |
| missing_values        |   80      |    0      |  -80      |
| duplicates            |   30      |    0      |  -30      |
| outliers_iqr          |   97      |    0      |  -97      |
| mean_text_length      | 1426.87   | 1223.02   | -203.85   |
| mean_rating           |    2.95   |    2.96   |    0.01   |
| label_imbalance_ratio |    0.0189 |    0.0195 |    0.0006 |
| unique_values_label   |    3      |    3      |    0      |

## Strategy B: drop + clip_zscore

| metric                |    before |     after |    change |
|:----------------------|----------:|----------:|----------:|
| total_rows            | 1040      |  932      | -108      |
| missing_values        |   80      |    0      |  -80      |
| duplicates            |   30      |    0      |  -30      |
| outliers_iqr          |   97      |   92      |   -5      |
| mean_text_length      | 1426.87   | 1376.9    |  -49.97   |
| mean_rating           |    2.95   |    2.96   |    0.01   |
| label_imbalance_ratio |    0.0189 |    0.0212 |    0.0023 |
| unique_values_label   |    3      |    3      |    0      |