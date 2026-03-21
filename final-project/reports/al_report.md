# Active Learning Report

## Experiment Setup
- **Dataset**: 1311 labeled examples (train: 1048, test: 263)
- **Model**: LogisticRegression (TF-IDF + balanced class weights)
- **Initial labeled**: 50 examples
- **Batch size**: 20 examples per iteration
- **Iterations**: 5

## Strategy Comparison

| Strategy | Final Accuracy | Final F1 |
|----------|---------------|----------|
| entropy | 0.7148 | 0.6834 |
| margin | 0.7148 | 0.6834 |
| least_confidence | 0.7148 | 0.6834 |
| random | 0.7034 | 0.6669 |

### Observations
- All uncertainty-based strategies (entropy, margin, least_confidence) perform identically
  for binary classification — this is expected since they are mathematically equivalent
  with 2 classes
- Uncertainty sampling outperforms random by ~1.1% accuracy and ~1.7% F1
- Most improvement happens in early iterations (50→90 examples)

## Final Model (Full Dataset)

- **Accuracy**: 0.7529
- **F1 (weighted)**: 0.7575
- **Training examples**: 1048

### Classification Report


## Files Generated
- 
- 
- 
- 
- 
