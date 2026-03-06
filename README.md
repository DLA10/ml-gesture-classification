# ML Gesture Phase Segmentation — Classification Coursework

MSc Data Science | Machine Learning Group Submission

---

## Notebook

| Notebook | Description |
|---|---|
| `MLCW_1_Simplified.ipynb` | Clean, compact ML classification notebook — all 11 models, full evaluation, and comparative analysis |

---

## Dataset

**OpenML ID:** 4538 — *Gesture Phase Segmentation*

- **Samples:** 9,873
- **Features:** 32 continuous (skeletal joint positions extracted from video recordings)
- **Classes:** 5 gesture phases:
  - `D` — Rest (default/idle position)
  - `H` — Hold (gesture held in place)
  - `P` — Preparation (movement towards gesture)
  - `R` — Retraction (return to rest)
  - `S` — Stroke (peak expressive phase)
- **Source:** Madeo et al. (2013), loaded via `sklearn.datasets.fetch_openml(data_id=4538)`
- **Class imbalance:** D and S are majority classes; H and R are minorities — balanced accuracy is used as the primary metric

---

## Classification Methods

### 1. Support Vector Machine (RBF Kernel)
SVM finds the maximum-margin hyperplane separating classes in a high-dimensional feature space. The RBF (Radial Basis Function) kernel maps data into an infinite-dimensional space, enabling non-linear decision boundaries. It is effective on mid-sized datasets with well-separated classes.
- **Key hyperparameters:** `C` (regularisation strength), `gamma` (kernel bandwidth)
- **Tuning:** RandomizedSearchCV, 25 iterations, 3-fold CV
- **Preprocessing:** StandardScaler inside Pipeline

### 2. Random Forest
An ensemble of decision trees, each trained on a bootstrapped subset of data with random feature selection at each split. Predictions are made by majority vote. Reduces variance through averaging while maintaining low bias.
- **Key hyperparameters:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Tuning:** RandomizedSearchCV, 40 iterations, 5-fold CV

### 3. K-Nearest Neighbours (KNN)
A non-parametric instance-based classifier that assigns a class based on the majority vote (or distance-weighted vote) of the k nearest training samples. Simple but computationally expensive at inference time. Sensitive to feature scale.
- **Key hyperparameters:** `n_neighbors` (k), `weights` (uniform/distance), `metric` (euclidean/manhattan)
- **Tuning:** GridSearchCV, 54 combinations, 5-fold CV
- **Preprocessing:** StandardScaler inside Pipeline

### 4. LightGBM
A gradient boosting framework using leaf-wise tree growth and histogram-based split finding. Significantly faster and more memory-efficient than traditional GBDT implementations. Excellent on tabular data.
- **Key hyperparameters:** `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `subsample`, `colsample_bytree`
- **Tuning:** RandomizedSearchCV, 40 iterations, 5-fold CV

### 5. Extra Trees (Extremely Randomised Trees)
Similar to Random Forest but selects split thresholds entirely at random rather than searching for the best threshold. This additional randomisation further reduces variance at the cost of slight bias, and is computationally faster.
- **Key hyperparameters:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Tuning:** RandomizedSearchCV, 40 iterations, 5-fold CV

### 6. Multi-Layer Perceptron (MLP)
A feedforward artificial neural network with one or more hidden layers. Trained via backpropagation with stochastic gradient descent. Capable of learning complex non-linear feature representations.
- **Key hyperparameters:** `hidden_layer_sizes` (architecture), `activation` (relu/tanh), `alpha` (L2 regularisation), `learning_rate`
- **Tuning:** RandomizedSearchCV, 30 iterations, 3-fold CV
- **Preprocessing:** StandardScaler inside Pipeline

### 7. XGBoost
A scalable gradient boosting framework using second-order gradient statistics and regularisation terms in the objective function. Builds trees sequentially, each correcting errors of the prior. Supports L1 and L2 regularisation natively and is highly competitive on tabular data.
- **Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- **Tuning:** GridSearchCV, 5-fold CV

### 8. Naive Bayes (Gaussian)
Applies Bayes' theorem under the assumption of conditional feature independence given the class label. Models each feature as a Gaussian distribution per class. Despite its strong independence assumption (violated by correlated skeletal features), it serves as a fast and interpretable baseline.
- **Key hyperparameter:** `var_smoothing` (adds fraction of max variance to all variances for numerical stability)
- **Tuning:** GridSearchCV, 11 values, 5-fold CV
- **Preprocessing:** StandardScaler inside Pipeline

### 9. Logistic Regression
A linear classifier that models the log-odds of class membership as a linear combination of features. Uses the multinomial (softmax) formulation for multi-class problems. The solver–penalty interaction (lbfgs only supports L2; saga supports L1 and L2) is handled via a list of parameter grids.
- **Key hyperparameters:** `C` (inverse regularisation), `penalty` (l1/l2), `solver` (lbfgs/saga)
- **Tuning:** GridSearchCV, 24 combinations, 5-fold CV
- **Preprocessing:** StandardScaler inside Pipeline

### 10. Gradient Boosting
A sequential ensemble method that fits each new tree to the residuals (negative gradient) of the combined model. Uses the scikit-learn implementation with a stage-wise additive approach. More interpretable than XGBoost but slower to train on large datasets.
- **Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- **Tuning:** GridSearchCV, 5-fold CV

### 11. Decision Tree
A single recursive tree that partitions the feature space using axis-aligned splits chosen to maximise information gain (Gini impurity). Fast and highly interpretable, but prone to overfitting without depth constraints. Serves as a baseline for the ensemble methods.
- **Key hyperparameters:** `max_depth`, `min_samples_split`, `min_samples_leaf`
- **Tuning:** GridSearchCV, 5-fold CV

---

## Evaluation Metrics

All models are evaluated on the held-out test set (30%) using the following metrics:

| Metric | Description |
|---|---|
| **Balanced Accuracy** | Mean per-class recall — preferred over raw accuracy due to class imbalance |
| **Macro ROC AUC** | Average One-vs-Rest AUC across all 5 classes, treating each equally |
| **Micro ROC AUC** | Aggregate OvR AUC across all samples — dominated by majority classes |
| **Precision** | Proportion of positive predictions that are correct (per class and macro/weighted average) |
| **Recall** | Proportion of actual positives correctly identified (per class and macro/weighted average) |
| **F1-Score** | Harmonic mean of precision and recall (per class and macro/weighted average) |
| **Classification Report** | Full per-class breakdown of precision, recall, F1, and support |

> **Why balanced accuracy?** The dataset has moderate class imbalance (D and S are more frequent than H and R). Raw accuracy would be misleading — a model that ignores minority classes can still appear accurate. Balanced accuracy penalises this equally across all classes.

---

## Visualisations

Each model produces the following plots:

### Confusion Matrix
Shows the count of correct and incorrect predictions for each class pair. Plotted using `ConfusionMatrixDisplay` with a blue colour map. Reveals which gesture phases are most commonly confused with each other.

### Per-Class OvR ROC Curves (per model)
One ROC curve per class using a One-vs-Rest (OvR) strategy. The curve plots True Positive Rate vs. False Positive Rate at varying classification thresholds. The area under each curve (AUC) measures how well the model distinguishes that class from all others. A perfect classifier has AUC = 1.0; a random classifier has AUC = 0.5.

### Final Comparison Bar Charts
Three side-by-side horizontal bar charts (sorted by Macro ROC AUC) comparing all 11 models across:
- Balanced Accuracy
- Macro OvR ROC AUC
- Micro OvR ROC AUC

### Per-Class ROC Comparison (all models)
One plot per class (5 total), with all 11 models overlaid on the same axes. Allows direct comparison of how well each model distinguishes a specific gesture phase from the rest.

---

## Experimental Protocol

- **Train/test split:** 70% training, 30% test — stratified by class, `random_state=42`
- **Hyperparameter tuning:** Performed on training data only (no leakage into test set)
- **Scoring metric:** `balanced_accuracy` for all `GridSearchCV` / `RandomizedSearchCV` calls
- **Preprocessing:** `StandardScaler` applied inside `Pipeline` for scale-sensitive models (SVM, KNN, MLP, Naive Bayes, Logistic Regression)
- **Tree-based models** (Random Forest, Extra Trees, LightGBM, XGBoost, Gradient Boosting, Decision Tree) require no feature scaling

---

## Requirements

```
scikit-learn
lightgbm
xgboost
numpy
pandas
matplotlib
seaborn
```

Install with:

```bash
pip install scikit-learn lightgbm xgboost numpy pandas matplotlib seaborn
```

---

## References

- Madeo, R. C. B., Lima, C. A. M., & Peres, S. M. (2013). *Gesture Unit Segmentation using Support Vector Machines.* ACM SAC.
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics.
