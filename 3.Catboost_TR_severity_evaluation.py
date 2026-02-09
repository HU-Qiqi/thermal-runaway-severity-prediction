# ============================================================
# CatBoost Classification and Error Analysis for TR Severity
# ------------------------------------------------------------
# This standalone script trains a CatBoost multiclass classifier
# for thermal runaway (TR) severity prediction and performs a
# detailed post-hoc error analysis.
#
# Key features:
#   - Native categorical feature handling with CatBoost
#   - Class-weighted training for imbalanced severity classes
#   - Evaluation using Accuracy, Macro-F1, and High-severity Recall
#   - Visualization of prediction–ground-truth relationships
#     using stacked bar charts (absolute counts and proportions)
#
# Input requirement:
#   - data/dataframe_processed.xlsx
#
# Output:
#   - Printed evaluation metrics
#   - Publication-ready figures saved to imgs/3/
# ============================================================

# =========================
# 0. Imports & Settings
# =========================
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

# =========================
# 1. Load Dataset
# =========================
print("Loading processed dataset...")

data_path = "data/dataframe_processed.xlsx"
df = pd.read_excel(data_path)

X = df.drop(columns=["Severity"])
y = df["Severity"]  # string labels: Low / Medium / High

severity_order = ["Low", "Medium", "High"]

# =========================
# 2. Train / Test Split
# =========================
X_cb = X.copy()

# Ensure categorical features are strings
CATEGORICAL_COLS = [
    "Cell-Description",
    "Manufacturer",
    "Geometry",
    "Trigger-Mechanism",
    "Bottom-Vent-Yes-No"
]

for col in CATEGORICAL_COLS:
    X_cb[col] = X_cb[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X_cb,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# =========================
# 3. Class Weights
# =========================
class_counter = Counter(y_train)
classes = sorted(class_counter.keys())

total_samples = len(y_train)
n_classes = len(classes)

class_weights = [
    total_samples / (n_classes * class_counter[c]) for c in classes
]

print("\nTraining set class distribution:", class_counter)
print("Class order:", classes)
print("Class weights:", class_weights)

# =========================
# 4. CatBoost Training
# =========================
cat_feature_indices = [X_cb.columns.get_loc(col) for col in CATEGORICAL_COLS]

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_feature_indices,
)

test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=cat_feature_indices,
)

cb_model = CatBoostClassifier(
    loss_function="MultiClass",
    eval_metric="TotalF1",
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5.0,
    iterations=1500,
    random_seed=42,
    od_type="Iter",
    od_wait=80,
    class_weights=class_weights,
    verbose=False,
)

print("\nTraining CatBoost model...")
cb_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# =========================
# 5. Evaluation Metrics
# =========================
y_pred = cb_model.predict(test_pool)
y_pred = np.asarray(y_pred).ravel()

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
high_recall = recall_score(
    y_test,
    y_pred,
    labels=["High"],
    average="macro",
    zero_division=0,
)

print("\nCatBoost Performance:")
print(f"Accuracy    : {acc:.3f}")
print(f"Macro-F1    : {f1_macro:.3f}")
print(f"High Recall : {high_recall:.3f}")

# =========================
# 6. Prediction vs True Label Analysis
# =========================
os.makedirs("imgs/3", exist_ok=True)

# Assemble prediction DataFrame
df_pred = pd.DataFrame({
    "True": y_test,
    "Predicted": y_pred,
})

# Enforce severity order
df_pred["True"] = pd.Categorical(df_pred["True"], categories=severity_order)
df_pred["Predicted"] = pd.Categorical(df_pred["Predicted"], categories=severity_order)#ordered=True

# Confusion-style count table: rows = Predicted, columns = True
count_table = pd.crosstab(df_pred["Predicted"], df_pred["True"])
count_table = count_table.reindex(index=severity_order, columns=severity_order, fill_value=0)

print("\nPrediction vs True Count Table:")
print(count_table)

# =========================
# 7. Visualization Settings
# =========================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.0,
})

risk_colors = ["#4C9F70",      # muted green (safe, stable)
               "#D4B483",   # desaturated sand / khaki
             "#C94A4A"     # muted red (hazard, not aggressive)
]

# =========================
# 8. Absolute Count Plot
# =========================
fig, ax = plt.subplots(figsize=(7, 5))

count_table.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=risk_colors,
    edgecolor="black",
    linewidth=1.0,
)

ax.set_xlabel("Predicted severity")
ax.set_ylabel("Number of samples")
ax.set_xticklabels(severity_order, rotation=0)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)


ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False,
)

plt.tight_layout()
plt.savefig("imgs/3/catboost_pred_vs_true_counts.png", dpi=600, bbox_inches="tight")
plt.show()

# =========================
# 9. Normalized Proportion Plot
# =========================
count_table_norm = count_table.div(count_table.sum(axis=1), axis=0).fillna(0)

fig, ax = plt.subplots(figsize=(7, 5))

count_table_norm.plot(
    kind="bar",
    stacked=True,
    ax=ax,
    color=risk_colors,
    edgecolor="black",
    linewidth=1.0,
)

ax.set_xlabel("Predicted severity")
ax.set_ylabel("Proportion within predicted class")
ax.set_xticklabels(severity_order, rotation=0)
ax.set_ylim(0, 1.1)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False,
)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig("imgs/3/catboost_pred_vs_true_normalized.png", dpi=600, bbox_inches="tight")
plt.show()
