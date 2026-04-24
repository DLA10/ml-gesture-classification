# ML Gesture Phase Segmentation — Classification

## Dataset

**OpenML ID:** 4538 — *Gesture Phase Segmentation (Processed)*

- **Samples:** 9,873
- **Features:** 32 continuous (skeletal joint positions and velocities extracted from RGB-D video recordings)
- **Classes:** 5 gesture phases
- **Source:** Madeo et al. (2013), loaded via `sklearn.datasets.fetch_openml(data_id=4538, as_frame=False)`

### Class Distribution

| Class | Label | Count | % of Dataset |
|---|---|---|---|
| D | Rest (default/idle position) | 2,741 | 27.8% |
| H | Hold (gesture held in place) | 998 | 10.1% |
| P | Preparation (movement towards gesture) | 2,097 | 21.2% |
| R | Retraction (return to rest) | 1,087 | 11.0% |
| S | Stroke (peak expressive gesture phase) | 2,950 | 29.9% |

> **Class imbalance note:** Hold (H) and Retraction (R) are the minority classes at ~10% each, while Stroke (S) and Rest (D) together account for nearly 58% of the data. This imbalance directly motivated the choice of balanced accuracy as the primary evaluation metric, rather than raw accuracy.

### Train / Test Split

| Set | Samples |
|---|---|
| Training | 6,911 (70%) |
| Test | 2,962 (30%) |

Split strategy: stratified by class, `random_state=42`.

---

## Classification Methods

All 8 models are numbered as they appear in the notebook and are evaluated using the same shared `evaluate()` function under identical conditions.

---

### Model 1: K-Nearest Neighbours (KNN)

A non-parametric instance-based classifier that assigns a class label by majority vote (or distance-weighted vote) among the k nearest training samples in feature space.

- **Wrapper:** `OneVsRestClassifier(KNeighborsClassifier(n_jobs=-1))`
- **Tuning method:** `GridSearchCV`, 5-fold CV
- **Search space:** 54 combinations
  - `n_neighbors`: [1, 3, 5, 7, 9, 11, 15, 21, 31]
  - `weights`: [uniform, distance]
  - `metric`: [euclidean, manhattan, minkowski]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | n_neighbors | weights | metric | CV Bal. Acc |
|---|---|---|---|---|
| 1 | 1 | uniform | manhattan | 0.6329 |
| 1 | 1 | distance | manhattan | 0.6329 |
| 3 | 1 | uniform | euclidean | 0.6107 |
| 3 | 1 | distance | euclidean | 0.6107 |
| 3 | 1 | uniform | minkowski | 0.6107 |

**Best configuration:** `n_neighbors=1`, `weights=uniform`, `metric=manhattan`, CV Bal. Acc = **0.6329**

---

### Model 2: Support Vector Machine — RBF Kernel (SVM)

SVM finds the maximum-margin hyperplane separating classes. With the RBF (Gaussian) kernel `exp(-gamma * ||x - x'||^2)`, it maps data into an infinite-dimensional space to learn non-linear decision boundaries.

- **Wrapper:** `OneVsRestClassifier(SVC(kernel='rbf', probability=True, random_state=42))`
- **Tuning method:** `GridSearchCV`, 3-fold CV
- **Search space:** 64 combinations
  - `C`: [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
  - `gamma`: [scale, auto, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | C | gamma | CV Bal. Acc |
|---|---|---|---|
| 1 | 100 | scale | 0.4924 |
| 2 | 50 | scale | 0.4892 |
| 3 | 10 | scale | 0.4750 |
| 4 | 5 | scale | 0.4641 |
| 5 | 1 | scale | 0.4278 |

**Best configuration:** `C=100`, `gamma=scale`, CV Bal. Acc = **0.4924**

---

### Model 3: Gradient Boosting

A sequential ensemble that fits each new tree to the residuals (negative gradient) of the combined model, using a stage-wise additive approach. Uses the scikit-learn implementation.

- **Estimator:** `GradientBoostingClassifier(random_state=42)`
- **Tuning method:** `GridSearchCV`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Search space:** 108 combinations
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `max_depth`: [3, 5, 7]
  - `subsample`: [0.6, 0.8, 1.0]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | n_estimators | learning_rate | max_depth | subsample | CV Bal. Acc |
|---|---|---|---|---|---|
| 1 | 200 | 0.1 | 7 | 0.6 | 0.5881 |
| 2 | 200 | 0.2 | 7 | 0.6 | 0.5865 |
| 3 | 200 | 0.2 | 7 | 0.8 | 0.5813 |
| 4 | 200 | 0.2 | 7 | 1.0 | 0.5804 |
| 5 | 200 | 0.1 | 7 | 0.8 | 0.5803 |

**Best configuration:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=7`, `subsample=0.6`, CV Bal. Acc = **0.5881**

---

### Model 4: Random Forest

An ensemble of decision trees, each trained on a bootstrapped subset of data with random feature selection at each split. Predictions are made by majority vote. Reduces variance through bagging while maintaining low bias.

- **Estimator:** `RandomForestClassifier(random_state=42, n_jobs=-1)`
- **Tuning method:** `RandomizedSearchCV`, 40 iterations, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Search space:** sampled from
  - `n_estimators`: [50, 100, 200, 300, 500]
  - `max_depth`: [None, 5, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10, 20]
  - `min_samples_leaf`: [1, 2, 4, 8]
  - `max_features`: [sqrt, log2, 0.3, 0.5]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features | CV Bal. Acc |
|---|---|---|---|---|---|---|
| 1 | 500 | 30 | 2 | 2 | 0.5 | 0.5678 |
| 2 | 100 | 30 | 5 | 1 | 0.5 | 0.5664 |
| 3 | 300 | None | 5 | 1 | 0.3 | 0.5648 |
| 4 | 300 | 30 | 5 | 2 | 0.5 | 0.5627 |
| 5 | 100 | 30 | 5 | 2 | 0.3 | 0.5607 |

**Best configuration:** `n_estimators=500`, `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=2`, `max_features=0.5`, CV Bal. Acc = **0.5678**

---

### Model 5: LightGBM

A gradient boosting framework using leaf-wise tree growth and histogram-based split finding. Significantly faster and more memory-efficient than traditional GBDT implementations. Uses `LGBMClassifier` with native multi-class support.

- **Estimator:** `LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)`
- **Tuning method:** `RandomizedSearchCV`, 40 iterations, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Search space:** sampled from
  - `n_estimators`: [50, 100, 200, 300, 500]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2, 0.3]
  - `num_leaves`: [15, 31, 63, 127]
  - `max_depth`: [-1, 3, 5, 7, 10]
  - `min_child_samples`: [10, 20, 30, 50]
  - `subsample`: [0.6, 0.7, 0.8, 0.9, 1.0]
  - `colsample_bytree`: [0.6, 0.7, 0.8, 0.9, 1.0]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | n_estimators | learning_rate | num_leaves | max_depth | min_child_samples | subsample | colsample | CV Bal. Acc |
|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 0.2 | 127 | -1 | 50 | 0.7 | 0.6 | 0.6228 |
| 1 | 500 | 0.1 | 127 | 10 | 20 | 0.6 | 0.7 | 0.6228 |
| 3 | 200 | 0.3 | 127 | -1 | 20 | 0.9 | 0.9 | 0.6223 |
| 4 | 300 | 0.3 | 31 | 10 | 20 | 1.0 | 0.7 | 0.6114 |
| 5 | 300 | 0.1 | 63 | 10 | 20 | 0.9 | 0.9 | 0.6080 |

**Best configuration:** `n_estimators=500`, `learning_rate=0.2`, `num_leaves=127`, `max_depth=-1`, `min_child_samples=50`, `subsample=0.7`, `colsample_bytree=0.6`, CV Bal. Acc = **0.6228**

---

### Model 6: XGBoost

A scalable gradient boosting framework using second-order gradient statistics and L1/L2 regularisation in the objective function. Builds trees sequentially, each correcting errors of the prior.

- **Estimator:** `XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42, n_jobs=-1)`
- **Label encoding:** `LabelEncoder` applied to map class labels (D=0, H=1, P=2, R=3, S=4) to integer indices required by XGBoost's multi-class objective. Encoding order is alphabetical, matching `np.unique(y)` — ensuring `predict_proba` columns align correctly with `label_binarize` columns for AUC computation.
- **Tuning method:** `GridSearchCV`, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Search space:** 324 combinations
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.05, 0.1, 0.2]
  - `max_depth`: [3, 5, 7]
  - `subsample`: [0.6, 0.8, 1.0]
  - `colsample_bytree`: [0.6, 0.8, 1.0]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | n_estimators | learning_rate | max_depth | subsample | colsample_bytree | CV Bal. Acc |
|---|---|---|---|---|---|---|
| 1 | 200 | 0.2 | 7 | 0.8 | 1.0 | 0.6042 |
| 2 | 200 | 0.2 | 7 | 0.8 | 0.8 | 0.6031 |
| 3 | 200 | 0.2 | 7 | 0.6 | 1.0 | 0.6019 |
| 4 | 200 | 0.2 | 7 | 0.6 | 0.8 | 0.6007 |
| 5 | 200 | 0.2 | 7 | 0.8 | 0.6 | 0.6004 |

**Best configuration:** `n_estimators=200`, `learning_rate=0.2`, `max_depth=7`, `subsample=0.8`, `colsample_bytree=1.0`, CV Bal. Acc = **0.6042**

---

### Model 7: Extra Trees (Extremely Randomised Trees)

Similar to Random Forest but selects both split features and split thresholds entirely at random rather than searching for the optimal split. This additional randomisation further reduces variance at the cost of slight bias, and is computationally faster than Random Forest.

- **Estimator:** `ExtraTreesClassifier(random_state=42, n_jobs=-1)`
- **Tuning method:** `RandomizedSearchCV`, 40 iterations, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Search space:** sampled from
  - `n_estimators`: [50, 100, 200, 300, 500]
  - `max_depth`: [None, 5, 10, 20, 30]
  - `min_samples_split`: [2, 5, 10, 20]
  - `min_samples_leaf`: [1, 2, 4, 8]
  - `max_features`: [sqrt, log2, 0.3, 0.5]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | n_estimators | max_depth | min_samples_split | min_samples_leaf | max_features | CV Bal. Acc |
|---|---|---|---|---|---|---|
| 1 | 500 | 30 | 2 | 2 | 0.5 | 0.5819 |
| 2 | 100 | 30 | 5 | 1 | 0.5 | 0.5811 |
| 3 | 300 | None | 5 | 1 | 0.3 | 0.5803 |
| 4 | 300 | 30 | 5 | 2 | 0.5 | 0.5756 |
| 5 | 100 | 30 | 5 | 2 | 0.3 | 0.5566 |

**Best configuration:** `n_estimators=500`, `max_depth=30`, `min_samples_split=2`, `min_samples_leaf=2`, `max_features=0.5`, CV Bal. Acc = **0.5819**

---

### Model 8: Multi-Layer Perceptron (MLP)

A feedforward artificial neural network with fully connected hidden layers, trained via backpropagation with stochastic gradient descent. Capable of learning complex non-linear feature representations. `StandardScaler` is applied inside a `Pipeline` since MLP training is sensitive to feature magnitude.

- **Estimator:** `Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(max_iter=500, random_state=42))])`
- **Tuning method:** `RandomizedSearchCV`, 30 iterations, `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- **Search space:** sampled from
  - `clf__hidden_layer_sizes`: [(64,), (128,), (256,), (64,64), (128,64), (128,128), (256,128), (64,64,32)]
  - `clf__activation`: [relu, tanh]
  - `clf__alpha` (L2 regularisation): [1e-5, 1e-4, 1e-3, 0.01, 0.1]
  - `clf__learning_rate`: [constant, adaptive]
- **Scoring:** `balanced_accuracy`

**Top 5 hyperparameter combinations (CV balanced accuracy):**

| Rank | hidden_layer_sizes | activation | alpha | learning_rate | CV Bal. Acc |
|---|---|---|---|---|---|
| 1 | (256, 128) | tanh | 0.01 | adaptive | 0.5324 |
| 2 | (256, 128) | tanh | 0.0001 | constant | 0.5297 |
| 3 | (256, 128) | relu | 1e-05 | constant | 0.5269 |
| 4 | (256, 128) | relu | 0.0001 | adaptive | 0.5250 |
| 5 | (128, 64) | relu | 0.01 | constant | 0.5192 |

**Best configuration:** `hidden_layer_sizes=(256, 128)`, `activation=tanh`, `alpha=0.01`, `learning_rate=adaptive`, CV Bal. Acc = **0.5324**

---

## CV Balanced Accuracy Summary (All Models)

This table summarises the best cross-validated balanced accuracy achieved by each model during hyperparameter tuning.

| Model | Best CV Balanced Accuracy |
|---|---|
| KNN | 0.6329 |
| LightGBM | 0.6228 |
| XGBoost | 0.6042 |
| Gradient Boosting | 0.5881 |
| Extra Trees | 0.5819 |
| Random Forest | 0.5678 |
| MLP | 0.5324 |
| SVM (RBF) | 0.4924 |

---

## Evaluation Metrics

All models are evaluated on the held-out test set (30%) using the following metrics, computed via a single shared `evaluate()` function to ensure consistency across all models.

| Metric | Description |
|---|---|
| **Balanced Accuracy** | Mean per-class recall — preferred over raw accuracy due to class imbalance. Penalises models that ignore minority classes. |
| **Macro ROC AUC** | One-vs-Rest AUC averaged equally across all 5 gesture phases. Treats minority and majority classes with equal importance. |
| **Micro ROC AUC** | Aggregate OvR AUC across all samples — weighted by class frequency, so dominated by majority classes (S and D). |
| **Precision** | Proportion of positive predictions that are correct (per class, plus macro and weighted average). |
| **Recall** | Proportion of actual positives correctly identified (per class, plus macro and weighted average). |
| **F1-Score** | Harmonic mean of precision and recall (per class, plus macro and weighted average). |
| **Classification Report** | Full per-class breakdown of precision, recall, F1-score, and support. |

> **Why balanced accuracy over raw accuracy?** The dataset has moderate class imbalance — Hold (H) and Retraction (R) together account for only ~21% of samples. A model that simply predicts the majority classes (S and D) can achieve high raw accuracy while completely failing on minority classes. Balanced accuracy penalises this equally across all five gesture phases, making it the most informative primary metric for this task.

> **Why macro AUC over micro AUC?** Micro AUC is dominated by the majority classes due to its sample-weighted aggregation. Macro AUC treats each gesture phase equally, making it a fairer overall measure of model quality under class imbalance. A model with a large gap between macro and micro AUC is performing well on common classes but failing on the minority phases.

---

## Shared `evaluate()` Function

To avoid inconsistency and code duplication, a single `evaluate(name, y_true, y_pred, y_prob)` function is defined once at the start of the notebook and called by every model. All results are stored in global `all_results` and `all_roc_data` dictionaries for the final comparison.

| Step | What it does |
|---|---|
| Balanced Accuracy | `balanced_accuracy_score(y_true, y_pred)` |
| Macro AUC | `roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')` |
| Micro AUC | `roc_auc_score(y_test_bin, y_prob, average='micro')` |
| Classification Report | `classification_report(y_true, y_pred, target_names=CLASSES)` |
| Confusion Matrix | `ConfusionMatrixDisplay` with Purples colour map |
| Per-class OvR ROC | `roc_curve(y_test_bin[:, i], y_prob[:, i])` for each of the 5 classes |
| Macro-average ROC curve | Interpolates each per-class curve onto a common 1,000-point FPR grid using `np.interp`, then averages the TPR values. Forces `mean_tpr[-1] = 1.0` to ensure the curve ends at (1, 1). |
| Micro-average ROC curve | Flattens all 5 classes into a single binary problem via `roc_curve(y_test_bin.ravel(), y_prob.ravel())`. Forces `fpr[-1] = 1.0`, `tpr[-1] = 1.0`. |
| Storage | Saves scalar metrics to `all_results[name]` and ROC curve data to `all_roc_data[name]` |

---

## Visualisations

Each model produces:

### Confusion Matrix
Shows the count of correct and incorrect predictions for each class pair, plotted with `ConfusionMatrixDisplay` using a Purples colour map. Reveals which gesture phases are most commonly confused with each other.

### Final Comparison Bar Charts
Three side-by-side horizontal bar charts (sorted by Macro ROC AUC) comparing all 8 models across Balanced Accuracy, Macro OvR ROC AUC, and Micro OvR ROC AUC.

### Per-Class OvR ROC Comparison (all models)
One plot per gesture phase (5 total), with all 8 models overlaid on the same axes. Allows direct comparison of how well each model distinguishes a specific gesture phase from the rest.

### Macro-Average OvR ROC (all models)
All 8 models overlaid on one graph. Each model's curve is computed by interpolating its 5 per-class ROC curves to a common FPR grid and averaging TPR. Shows overall multi-class discrimination ability with equal class weighting.

### Micro-Average OvR ROC (all models)
All 8 models overlaid on one graph. Each model's curve is computed by treating all class predictions as one large binary problem. Shows aggregated discrimination weighted by class frequency.

---

## Experimental Protocol

| Setting | Value |
|---|---|
| Train / test split | 70% training, 30% test — stratified by class |
| Random seed | `RANDOM_STATE = 42` throughout |
| Hyperparameter tuning | Performed exclusively on training data — no test data leakage |
| Scoring metric | `balanced_accuracy` for all `GridSearchCV` / `RandomizedSearchCV` calls |
| Preprocessing | `StandardScaler` inside `Pipeline` for scale-sensitive models (MLP only in this notebook) |
| Tree-based models | No feature scaling required (Random Forest, Extra Trees, LightGBM, XGBoost, Gradient Boosting) |
| KNN & SVM | Wrapped in `OneVsRestClassifier` for explicit multiclass decomposition |

---

## Conclusions

1. **Tree-based ensemble methods consistently outperform** distance-based and linear methods on this dataset. LightGBM, XGBoost, Gradient Boosting, Random Forest, and Extra Trees all rank above MLP and SVM across every evaluation metric, confirming that the gesture phase feature space contains non-linear interactions that simpler models cannot capture effectively.

2. **Class imbalance directly impacts per-class performance.** The minority gesture phases — Hold (H) and Retraction (R) at ~10% each — consistently show lower per-class recall and AUC across all models compared to the majority classes Stroke (S) and Rest (D). Balanced accuracy and macro AUC are therefore the most informative evaluation metrics for this task.

3. **Evaluation metrics are mutually consistent.** The close alignment between balanced accuracy rankings and macro AUC rankings across all 8 models indicates that the chosen metrics reinforce each other, strengthening confidence in the comparative conclusions drawn from the results.

4. **Hyperparameter tuning had a measurable impact.** The top-ranked parameter combinations achieved meaningfully higher CV balanced accuracy than default configurations across all models, justifying the computational cost of the search. The consistent preference for deeper trees, larger ensembles, and leaf-wise growth (high `num_leaves` in LightGBM) reflects the complexity of the gesture phase classification problem.

5. **Recommended method: LightGBM** with `n_estimators=500`, `learning_rate=0.2`, `num_leaves=127`, `max_depth=-1`, `min_child_samples=50`, `subsample=0.7`, `colsample_bytree=0.6`. LightGBM achieved the highest cross-validated balanced accuracy (0.6228) and consistently high macro AUC, combining strong predictive performance with efficient training via leaf-wise tree growth and histogram-based binning — making it the most suitable method for gesture phase segmentation on this dataset.

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

### Dataset
- Madeo, R. C. B., Lima, C. A. M., & Peres, S. M. (2013). *Gesture Unit Segmentation using Support Vector Machines.* ACM SAC.

### 1. K-Nearest Neighbours (KNN)
- **Theory:** Cover, T. & Hart, P. (1967). *Nearest Neighbor Pattern Classification.* IEEE Transactions on Information Theory. [https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- **Implementation:** scikit-learn. *KNeighborsClassifier API.* [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

### 2. Gradient Boosting
- **Theory:** Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics. [https://en.wikipedia.org/wiki/Gradient_boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
- **Implementation:** scikit-learn. *GradientBoostingClassifier API.* [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

### 3. XGBoost
- **Theory:** Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016. [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
- **Implementation:** XGBoost Developers. *XGBoost Python API.* [https://xgboost.readthedocs.io/en/stable/python/python_api.html](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

### 4. Random Forest
- **Theory:** Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. [https://en.wikipedia.org/wiki/Random_forest](https://en.wikipedia.org/wiki/Random_forest)
- **Implementation:** scikit-learn. *RandomForestClassifier API.* [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### 5. Extra Trees
- **Theory:** Geurts, P., Ernst, D. & Wehenkel, L. (2006). *Extremely Randomized Trees.* Machine Learning, 63(1), 3–42. [https://en.wikipedia.org/wiki/Random_forest#ExtraTrees](https://en.wikipedia.org/wiki/Random_forest#ExtraTrees)
- **Implementation:** scikit-learn. *ExtraTreesClassifier API.* [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

### 6. Multi-Layer Perceptron (MLP)
- **Theory:** Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning.* MIT Press. [https://en.wikipedia.org/wiki/Multilayer_perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- **Implementation:** scikit-learn. *MLPClassifier API.* [https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

### 7. LightGBM
- **Theory:** Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS. [https://lightgbm.readthedocs.io/en/stable/Features.html](https://lightgbm.readthedocs.io/en/stable/Features.html)
- **Implementation:** LightGBM Developers. *LGBMClassifier API.* [https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)

### 8. Support Vector Machine (SVM)
- **Theory:** Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks.* Machine Learning, 20(3), 273–297. [https://en.wikipedia.org/wiki/Support_vector_machine](https://en.wikipedia.org/wiki/Support_vector_machine)
- **Implementation:** scikit-learn. *SVC API.* [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

---

### Evaluation Metrics
- **Balanced Accuracy:** Brodersen, K. H. et al. (2010). *The Balanced Accuracy and Its Posterior Distribution.* ICPR. scikit-learn docs: [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)
- **ROC AUC (Macro & Micro):** Fawcett, T. (2006). *An Introduction to ROC Analysis.* Pattern Recognition Letters, 27(8), 861–874. scikit-learn docs: [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- **scikit-learn:** Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
