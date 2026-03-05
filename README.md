# ML Gesture Phase Segmentation — Classification Coursework

MSc Data Science | Machine Learning Group Submission

---

## Notebook

| Notebook | Description |
|---|---|
| `MLCW_1_Simplified.ipynb` | Clean, compact ML classification notebook — all 9 models, full evaluation, and comparative analysis |

---

## Dataset

**OpenML ID:** 4538 — *Gesture Phase Segmentation*

- **Samples:** 9,873
- **Features:** 32 continuous (skeletal joint positions from video)
- **Classes:** 5 — D (Rest), H (Hold), P (Preparation), R (Retraction), S (Stroke)
- **Source:** Madeo et al. (2013), loaded via `sklearn.datasets.fetch_openml`

---

## Models

All 9 classifiers are implemented with hyperparameter search, cross-validation, and full evaluation.

| # | Model | Search | CV | Folds |
|---|---|---|---|---|
| 1 | Support Vector Machine (RBF) | RandomizedSearchCV | StratifiedKFold | 3 |
| 2 | Random Forest | RandomizedSearchCV | StratifiedKFold | 5 |
| 3 | K-Nearest Neighbours | GridSearchCV | StratifiedKFold | 5 |
| 4 | LightGBM | RandomizedSearchCV | StratifiedKFold | 5 |
| 5 | Extra Trees | RandomizedSearchCV | StratifiedKFold | 5 |
| 6 | Multi-Layer Perceptron | RandomizedSearchCV | StratifiedKFold | 3 |
| 7 | Linear Discriminant Analysis | GridSearchCV | StratifiedKFold | 5 |
| 8 | Naive Bayes (Gaussian) | GridSearchCV | StratifiedKFold | 5 |
| 9 | Logistic Regression | GridSearchCV | StratifiedKFold | 5 |

---

## Protocol

- **Split:** 70% train / 30% test, stratified by class (`random_state=42`)
- **Scoring:** `balanced_accuracy` for all hyperparameter searches
- **Preprocessing:** `StandardScaler` inside `Pipeline` (where required) to prevent data leakage
- **Evaluation:** Balanced accuracy, macro/micro ROC AUC, classification report, confusion matrix, per-class OvR ROC curves

---

## Requirements

```
scikit-learn
lightgbm
numpy
pandas
matplotlib
seaborn
```

Install with:

```bash
pip install scikit-learn lightgbm numpy pandas matplotlib seaborn
```

---

## References

- Madeo, R. C. B., Lima, C. A. M., & Peres, S. M. (2013). *Gesture Unit Segmentation using Support Vector Machines.* ACM SAC.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
