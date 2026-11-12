# Credit Card Fraud Detection

Detecting fraudulent card transactions in a highly imbalanced dataset (about 0.17% fraud) with Logistic Regression, Random Forest and XGBoost, using SMOTE to balance the training data.

## Approach

1. **Data:** 284,807 European card transactions from September 2013. Features `V1`–`V28` are PCA components; `Time` and `Amount` are scaled.
2. **Split first, then resample:** an 80/20 stratified split, with SMOTE applied **only to the training set**, so the test set keeps the real class balance.
3. **Models:** Logistic Regression (baseline), Random Forest and XGBoost.
4. **Evaluation:** accuracy is misleading at 0.17% fraud, so the models are compared on precision, recall, F1-score, ROC-AUC, PR-AUC and the confusion matrix.

## Results

Run the script on the full dataset and paste the table it prints here:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

The confusion matrices are saved to `results/confusion_matrices.png`.

## Run it

```bash
pip install -r requirements.txt
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/
python fraud_detection.py --data data/creditcard.csv

# Quick demo without the dataset
python fraud_detection.py --synthetic
```

Options: `--no-smote` trains on the original class balance, for comparison.

## Tech

Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Matplotlib
