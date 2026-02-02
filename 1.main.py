# ============================================================
# Battery Thermal Runaway Severity Classification
# CatBoost Multiclass Model + SHAP Interpretability
# ------------------------------------------------------------
# This script:
#   1. Preprocesses FTRC battery failure data
#   2. Constructs a Severity Index (SI)
#   3. Trains a CatBoost multiclass classifier (Low / Medium / High)
#   4. Evaluates performance with cross-validation and test split
#   5. Performs SHAP-based global and class-specific interpretability
#   6. Saves trained models, metrics, and publication-quality figures
#
# NOTE:
#   - Code functionality and outputs are identical to the original version
#   - Comments, prints, and variable naming are standardized in English
#   - Redundant blocks are consolidated for readability
# ============================================================

# =========================
# 0. Imports & Settings
# =========================
import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

from catboost import CatBoostClassifier, Pool

from src.data import FTRC_Data

warnings.filterwarnings("ignore")

# =========================
# 1. Load and Filter Data
# =========================
print("Loading FTRC battery failure dataset...")
data_train = FTRC_Data()

# Keep only fully charged cells (SOC = 100%)
data_train.df = data_train.df[
    data_train.df["Pre-Test-State-of-Charge-%"] == 100
]

# =========================
# 2. Feature Definitions
# =========================
FEATURES_METADATA = [
    "Cell-Description",
    "Manufacturer",
    "Geometry",
    "Cell-Capacity-Ah",
    "Trigger-Mechanism",
    "BV Actuated",
]

FEATURES_EJECTED_MASS = [
    "Total-Mass-Ejected-g",
    "Total Ejected Mass Fraction [g/g]",
    "Post-Test-Mass-Unrecovered-g",
    "Unrecovered Mass Fraction [g/g]",
    "Pre-Test-Cell-Mass-g",
    "Post-Test-Mass-Cell-Body-g",
    "Body Mass Remaining Fraction [g/g]",
    "Positive-Mass-Ejected-g",
    "Positive Ejected Mass Fraction [g/g]",
    "Negative-Mass-Ejected-g",
    "Negative Ejected Mass Fraction [g/g]",
]

TARGETS_HEAT = [
    "Total Heat Output [kJ/A*h]",
    "Cell Body Heat Output [kJ/A*h]",
    "Positive Heat Output [kJ/A*h]",
    "Negative Heat Output [kJ/A*h]",
]

# Subset for modeling
data_trimmed = data_train.df[FEATURES_METADATA].copy()

# =========================
# 3. Severity Index (SI)
# =========================
print("Computing Severity Index (SI)...")

# Normalization (max-scaling)
data_train.df["Total Heat Output [kJ/A*h]"] /= data_train.df[
    "Total Heat Output [kJ/A*h]"
].max()

data_train.df["Total-Mass-Ejected-g"] /= data_train.df[
    "Total-Mass-Ejected-g"
].max()

data_train.df["Negative-Mass-Ejected-g"] /= data_train.df[
    "Negative-Mass-Ejected-g"
].max()

# Equal-weight severity index
w1 = w2 = w3 = w4 = 0.25

SI = (
    w1 * data_train.df["Total Heat Output [kJ/A*h]"]
    + w2 * data_train.df["Total-Mass-Ejected-g"]
    + w3 * data_train.df["Negative-Mass-Ejected-g"]
    + w4 * data_train.df["Total Ejected Mass Fraction [g/g]"]
)

# =========================
# 4. Severity Labeling
# =========================
print("Assigning severity labels using quantiles...")

quantile_bins = [0.0, 0.15, 0.85, 1.0]

data_trimmed["Severity"] = pd.qcut(
    SI,
    q=quantile_bins,
    labels=["Low", "Medium", "High"],
    duplicates="drop",
)

print("Severity class distribution:")
print(data_trimmed["Severity"].value_counts(normalize=True))

# =========================
# 5. Train/Test Split
# =========================
X = data_trimmed.drop(columns=["Severity"])
y = data_trimmed["Severity"]

# Save processed dataframe
os.makedirs("data", exist_ok=True)
X.assign(Severity=y).to_excel("data/dataframe_processed.xlsx", index=False)

# =========================
# 6. Categorical Feature Handling
# =========================
CATEGORICAL_COLS = [
    "Cell-Description",
    "Manufacturer",
    "Geometry",
    "Trigger-Mechanism",
    "BV Actuated",
]

for col in CATEGORICAL_COLS:
    X[col] = X[col].astype(str)

cat_feature_indices = [X.columns.get_loc(c) for c in CATEGORICAL_COLS]

# =========================
# 7. Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# =========================
# 8. Class Weights
# =========================
class_counter = Counter(y_train)
classes = sorted(class_counter.keys())

total_samples = len(y_train)
n_classes = len(classes)

class_weights = [
    total_samples / (n_classes * class_counter[c]) for c in classes
]

print("Training class distribution:", class_counter)
print("Class weights:", class_weights)

# =========================
# 9. Cross-Validation (5-fold)
# =========================
print("Running 5-fold stratified cross-validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
    train_pool_cv = Pool(
        X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_feature_indices
    )
    val_pool_cv = Pool(
        X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_feature_indices
    )

    model_cv = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=10,
        iterations=1500,
        random_seed=fold,
        od_type="Iter",
        od_wait=80,
        class_weights=class_weights,
        verbose=False,
    )

    model_cv.fit(train_pool_cv, eval_set=val_pool_cv, use_best_model=True)
    preds_val = model_cv.predict(val_pool_cv)
    acc_val = accuracy_score(y.iloc[val_idx], preds_val)
    cv_accuracies.append(acc_val)

    print(f"Fold {fold} accuracy: {acc_val:.4f}")

print("Mean CV accuracy:", np.mean(cv_accuracies))
print("CV accuracy std:", np.std(cv_accuracies))

# =========================
# 10. Final Model Training
# =========================
print("Training final CatBoost model...")

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

final_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="MultiClass",
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=10,
    iterations=1500,
    random_seed=42,
    od_type="Iter",
    od_wait=80,
    class_weights=class_weights,
    verbose=False,
)

final_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

y_pred = final_model.predict(test_pool)

# =========================
# 11. Evaluation
# =========================
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
high_recall = recall_score(
    y_test, y_pred, labels=["High"], average="macro", zero_division=0
)

print("Final Test Performance:")
print(f"Accuracy     : {acc:.3f}")
print(f"Macro-F1     : {f1_macro:.3f}")
print(f"High Recall  : {high_recall:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 12. Save Model & Metadata
# =========================
os.makedirs("models", exist_ok=True)

final_model.save_model("models/catboost_final_model.cbm")

with open("models/cv_metrics.json", "w") as f:
    json.dump(
        {
            "cv_accuracies": [float(a) for a in cv_accuracies],
            "mean_cv_accuracy": float(np.mean(cv_accuracies)),
            "std_cv_accuracy": float(np.std(cv_accuracies)),
        },
        f,
        indent=4,
    )

with open("models/model_metadata.json", "w") as f:
    json.dump(
        {
            "class_order": classes,
            "class_weights": class_weights,
            "categorical_features": CATEGORICAL_COLS,
        },
        f,
        indent=4,
    )

print("Model and metadata successfully saved.")

# ============================================================
# 13. SHAP Analysis & Visualization
# ============================================================
print("\n\nRunning SHAP analysis...")

# -------------------------
# 13.1 SHAP Value Computation
# -------------------------
train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_feature_indices,
)

# For multiclass models:
# SHAP shape = (n_samples, n_classes, n_features + bias)
shap_values = final_model.get_feature_importance(
    train_pool,
    type="ShapValues",
)

print("SHAP array shape:", shap_values.shape)

# Remove the last column (bias term)
shap_no_bias = shap_values[:, :, :-1]

# Mean absolute SHAP across samples and classes
mean_abs_shap = np.mean(np.abs(shap_no_bias), axis=(0, 1))

# Sort by importance
feature_names = X_train.columns.to_numpy()
idx_sorted = np.argsort(mean_abs_shap)[::-1]

sorted_importance = mean_abs_shap[idx_sorted]
sorted_features = feature_names[idx_sorted]

# -------------------------
# 13.2 Global SHAP Importance Plot
# -------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    range(len(sorted_importance)),
    sorted_importance[::-1],
    color="#4C72B0", 
    edgecolor="black",
    linewidth=1.0,
)

ax.set_yticks(range(len(sorted_importance)))
ax.set_yticklabels(sorted_features[::-1])
ax.set_xlabel("Mean |SHAP value| across samples and classes")

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig(
    "imgs/1/Global_Feature_Importance_(CatBoost_SHAP).png",
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# -------------------------
# 13.3 SHAP Importance for High-Severity Class
# -------------------------
print("\nComputing SHAP importance for High-severity class...")

class_order = ["High", "Low", "Medium"]
high_class_idx = class_order.index("High")

shap_high = shap_values[:, high_class_idx, :-1]
mean_abs_shap_high = np.mean(np.abs(shap_high), axis=0)

idx_sorted_high = np.argsort(mean_abs_shap_high)[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    range(len(idx_sorted_high)),
    mean_abs_shap_high[idx_sorted_high][::-1],
    color="#4C72B0",
    edgecolor="black",
    linewidth=1.0,
)

ax.set_yticks(range(len(idx_sorted_high)))
ax.set_yticklabels(feature_names[idx_sorted_high][::-1])
ax.set_xlabel("Mean |SHAP value| for class = High")

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig(
    "imgs/1/Feature_Importance_for_High-Severity_TR(CatBoost_SHAP).png",
    dpi=600,
    bbox_inches="tight",
)
plt.show()

# ============================================================
# 14. Model Performance Visualization
# ============================================================
print("\nGenerating performance figures...")

# -------------------------
# 14.1 Cross-Validation Accuracy
# -------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
})

folds = np.arange(1, len(cv_accuracies) + 1)
mean_acc = np.mean(cv_accuracies)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    folds,
    cv_accuracies,
    marker="o",
    markersize=8,
    linewidth=2.5,
    color="#4C72B0",
    markeredgecolor="black",
)
ax.axhline(
    mean_acc,
    linestyle="--",
    linewidth=2,
    color="#4C72B0",
    label=f"Mean = {mean_acc:.3f}",
)

ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_xticks(folds)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("imgs/1/catboost_cv_accuracies.png", dpi=600, bbox_inches="tight")
plt.show()

# -------------------------
# 14.2 Per-Class Metrics
# -------------------------
y_pred_flat = np.asarray(y_pred).ravel()

report = classification_report(y_test, y_pred_flat, output_dict=True)
report_df = pd.DataFrame(report).T

cls_metrics = report_df.loc[
    ["Low", "Medium", "High"],
    ["precision", "f1-score","recall",],
]

fig, ax = plt.subplots(figsize=(7, 5))
cls_metrics.plot(
    kind="bar",
    ax=ax,
    color=[ "#4C72B0","#8172B3","#55A868"],
    edgecolor="black",
    linewidth=1.0,
)

ax.set_xlabel("Severity")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticklabels(cls_metrics.index, rotation=0)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

ax.legend(
    ["Accuracy", "Macro-f1", "Recall"],  # 自定义文本
    loc="lower center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False,
)

plt.tight_layout()
plt.savefig("imgs/1/catboost_per_class_metrics.png", dpi=600, bbox_inches="tight")
plt.show()

# -------------------------
# 14.3 Native CatBoost Feature Importance
# -------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 18,
    "axes.linewidth": 20,
})

fi = final_model.get_feature_importance(train_pool)
fi_df = (
    pd.DataFrame({"feature": X.columns, "importance": fi})
    .sort_values("importance", ascending=False)
    .head(15)
)

sns.set_style("white")
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(
    data=fi_df,
    x="importance",
    y="feature",
    color="#4C72B0",
    edgecolor="black",
    linewidth=1.0,
    ax=ax,
)

ax.set_xlabel("CatBoost feature importance")
ax.set_ylabel("Feature")

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig("imgs/1/catboost_feature_importance.png", dpi=600, bbox_inches="tight")
plt.show()
