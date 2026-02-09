# ============================================================
# Benchmarking Classification Algorithms for TR Severity
# ------------------------------------------------------------
# This script compares the performance of multiple classification
# algorithms (Logistic Regression, Random Forest, XGBoost, and
# CatBoost) for thermal runaway (TR) severity prediction.
#
# Evaluation metrics:
#   - Accuracy
#   - Macro-F1 score
#   - Recall for the High-severity class (safety-critical)
#
# Notes:
#   - All sklearn/XGBoost models share identical preprocessing
#     (One-Hot Encoding + Standardization).
#   - CatBoost results are imported from a dedicated pipeline
#     using native categorical handling.
#   - This script is self-contained and independent from other
#     training scripts in the repository.
# ============================================================

# =========================
# 0. Imports & Settings
# =========================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================
# 1. Load Processed Dataset
# =========================
print("Loading processed dataset...")

df = pd.read_excel("data/dataframe_processed.xlsx")

X = df.drop(columns=["Severity"])
y = df["Severity"]

# Enforce ordered severity labels: Low < Medium < High
severity_order = ["Low", "Medium", "High"]
y_cat = pd.Categorical(y, categories=severity_order, ordered=True)

# Encode labels for sklearn / XGBoost
# Low=0, Medium=1, High=2
y_encoded = y_cat.codes

print("Label encoding: 0=Low, 1=Medium, 2=High")
print("Unique encoded labels:", np.unique(y_encoded))

# =========================
# 2. Train / Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

# Categorical features (consistent with CatBoost setup)

CATEGORICAL_COLS = [
    "Cell-Description",
    "Manufacturer",
    "Geometry",
    "Trigger-Mechanism",
    "Bottom-Vent-Yes-No"
]

NUMERIC_COLS = [c for c in X.columns if c not in CATEGORICAL_COLS]

# =========================
# 3. Preprocessing Pipeline
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERIC_COLS),
    ]
)

# =========================
# 4. Define Benchmark Models
# =========================
benchmark_models = {
    "LR": LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="lbfgs",
        C=2.0,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "RF": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        use_label_encoder=False,
    ),
}

# =========================
# 5. Train & Evaluate Models
# =========================
results = []

print("\nTraining and evaluating benchmark models...")

for name, model in benchmark_models.items():
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    high_recall = recall_score(
        y_test,
        y_pred,
        labels=[2],  # High severity
        average="macro",
        zero_division=0,
    )

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Macro-F1": f1_macro,
        "High Recall": high_recall,
    })

    print(f"{name}: Acc={acc:.3f}, Macro-F1={f1_macro:.3f}, High-Recall={high_recall:.3f}")

# =========================
# 6. CatBoost Reference Result
# =========================
# Obtained from a dedicated CatBoost pipeline using native
# categorical feature handling

results.append({
    "Model": "CatBoost",
    "Accuracy": 0.810,
    "Macro-F1": 0.792,
    "High Recall": 1.000,
})

print("CatBoost: Acc=0.810, Macro-F1=0.792, High-Recall=1.000")

# =========================
# 7. Results Summary
# =========================
results_df = pd.DataFrame(results)

print("\nOverall benchmark results:")
print(results_df.round(3))

# =========================
# 8. Visualization
# =========================
plot_df = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Macro-F1", "High Recall"],
    var_name="Metric",
    value_name="Score",
)

os.makedirs("imgs/2", exist_ok=True)

sns.set_style("white")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric",
    palette=[ "#4C72B0","#8172B3","#55A868"],
    edgecolor="black",
    linewidth=1.0,
    ax=ax,
)

ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)


legend = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=False,
    columnspacing=0.8
)

for text in legend.get_texts():
    text.set_text(text.get_text().capitalize())

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.2)

ax.yaxis.set_tick_params(left=True)

ax.tick_params(
    axis="y",
    which="major",
    length=4,      
    width=1.2,     
    direction="out"
)

plt.tight_layout()
plt.savefig("imgs/2/model_comparison_metrics_tuned.png", dpi=600, bbox_inches="tight")
plt.show()
