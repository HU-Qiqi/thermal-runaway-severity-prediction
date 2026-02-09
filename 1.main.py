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
import numpy as np

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


def cylinder_volume_L(code):
    code = int(code)
    
    diameter = code // 1000
    height = (code % 1000)/10
    
    radius = diameter / 2
    
    volume_mm3 = np.pi * (radius**2) * height
    
    return volume_mm3 / 1e6   # 转成 L

data_train.df["Cell_Volume_L"] = data_train.df["Geometry"].apply(cylinder_volume_L)
data_train.df["Volumetric-Energy-Density-Wh/L"] = data_train.df['Cell-Energy-Wh']/data_train.df["Cell_Volume_L"] 

data_train.df["Gravimetric-Energy-Density-Wh/kg"]=data_train.df['Cell-Energy-Wh']/data_train.df['Pre-Test-Cell-Mass-g']*1000



# =========================
# 2. Feature Definitions
# =========================
FEATURES_METADATA = [
    "Cell-Description",
    "Manufacturer",
    "Geometry",
    "Cell-Capacity-Ah",
    "Trigger-Mechanism", 
    "Bottom-Vent-Yes-No","Gravimetric-Energy-Density-Wh/kg","Volumetric-Energy-Density-Wh/L"
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
    "Bottom-Vent-Yes-No"
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



import numpy as np
import matplotlib.pyplot as plt

# shap_values: (n_samples, n_classes, n_features+1) -> 去掉最后一列
shap_per_class = shap_values[:, :, :-1]  # (N, C, F)

class_names = ["High", "Low", "Medium"]
feature_names = np.array(feature_names)

mean_abs = np.mean(np.abs(shap_per_class), axis=0)  # (C, F)
total = mean_abs.sum(axis=0)                        # (F,)
idx = np.argsort(total)[::-1]

top_k = 8
idx = idx[:top_k]

mean_abs_top = mean_abs[:, idx]     # (C, K)
features_top = feature_names[idx]   # (K,)

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
})

fig, ax = plt.subplots(figsize=(11, 8))

colors = [ "#C94A4A", "#4C9F70", "#D4B483"]  # High/Low/Medium

# ✅ 拉开行间距（更呼吸）
y = np.arange(top_k)[::-1] * 1.15
features_plot = features_top[::-1]
vals_plot = mean_abs_top[:, ::-1]

left = np.zeros(top_k)

for c, (cls, col) in enumerate(zip(class_names, colors)):
    ax.barh(
        y,
        vals_plot[c],
        left=left,
        height=0.65,          # ✅ 条形更细
        color=col,
        edgecolor="black",
        linewidth=0.6,        # ✅ 边框更柔和
        label=cls
    )
    left += vals_plot[c]

ax.set_yticks(y)
ax.set_yticklabels(features_plot)
ax.set_xlabel("Mean |SHAP value| by class")

    # legend 放图内
ax.legend(
        frameon=False,
        loc="upper right"
    )

# ✅ 更现代论文风：去掉上/右边框
ax.spines["top"].set_linewidth(1.2)
ax.spines["right"].set_linewidth(1.2)
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

plt.tight_layout()
plt.savefig("imgs/1/Global_SHAP_Stacked_By_Class.png", dpi=600, bbox_inches="tight")
plt.show()


# ============================================================
# 13.X  SHAP plot: categorical expanded by levels (grouped)
#       + numeric features as single-row stacked bars
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# shap_values: (N, C, F+1)  -> remove bias
shap_no_bias = shap_values[:, :, :-1]  # (N, C, F)

class_names = ["High", "Low", "Medium"]               # 与你模型一致
colors = [ "#C94A4A","#4C9F70", "#D4B483"]            # High/Low/Medium
feature_names = X_train.columns.to_numpy()

# 你框出来的类别型特征
categorical_features_to_expand = [
    "Manufacturer",
    "Cell-Description",
    "Trigger-Mechanism",
    "Geometry",
    "Bottom-Vent-Yes-No"
]

# 每个类别型特征最多展示多少个 level（防止图太长）
top_n_levels = 6

# ---------------------------
# 1) 收集条目：类别型（按level） + 数值型（每特征1行）
# ---------------------------
rows = []  # dict(label=..., shap=(C,), kind='cat'/'num', group=...)

# ---- A) 类别型：按 level 展开 ----
for feat in categorical_features_to_expand:
    if feat not in X_train.columns:
        continue

    feat_idx = np.where(feature_names == feat)[0][0]

    # 选取最常见的 levels（你也可以替换为按 SHAP 贡献选）
    vc = X_train[feat].astype(str).value_counts(dropna=False)
    levels = vc.index[:top_n_levels].tolist()

    for lv in levels:
        mask = (X_train[feat].astype(str).values == str(lv))
        if mask.sum() == 0:
            continue

        shap_sub = shap_no_bias[mask, :, feat_idx]          # (n_sub, C)
        mean_abs_sub = np.mean(np.abs(shap_sub), axis=0)    # (C,)

        rows.append({
        "label": str(lv),   # ← 只保留类别名称
        "shap": mean_abs_sub,
        "kind": "cat",
        "group": feat
        })

# ---- B) 数值型：每个特征只保留 1 行（整体均值）----
numeric_features = [c for c in X_train.columns if c not in categorical_features_to_expand]

for feat in numeric_features:
    feat_idx = np.where(feature_names == feat)[0][0]
    shap_feat = shap_no_bias[:, :, feat_idx]                 # (N, C)
    mean_abs_feat = np.mean(np.abs(shap_feat), axis=0)       # (C,)

    rows.append({
        "label": feat,
        "shap": mean_abs_feat,
        "kind": "num",
        "group": "Numeric"
    })

if len(rows) == 0:
    print("No SHAP entries to plot.")
else:
    # ---------------------------
    # 2) 按 group 分组排序：保证同一大特征的 level 连续
    #    - group 之间按 group 总重要性排序
    #    - group 内按条目重要性排序
    # ---------------------------
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    group_scores = {
        g: float(np.sum([it["shap"].sum() for it in items]))
        for g, items in groups.items()
    }

    # group 按重要性降序（你也可以固定顺序）
    group_order = sorted(group_scores.keys(), key=lambda g: group_scores[g], reverse=True)

    # （可选）让 Numeric 固定放最后（更像图2那种“类别块+数值块”）
    if "Numeric" in group_order:
        group_order = [g for g in group_order if g != "Numeric"] + ["Numeric"]

    rows_grouped = []
    for g in group_order:
        items_sorted = sorted(groups[g], key=lambda r: r["shap"].sum(), reverse=True)
        rows_grouped.extend(items_sorted)

    rows = rows_grouped

    # ---------------------------
    # 3) 画堆叠水平条形图
    # ---------------------------
    plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.0,
})


    fig, ax = plt.subplots(figsize=(13, 11))

    y = np.arange(len(rows))
    left = np.zeros(len(rows))

    shap_mat = np.vstack([r["shap"] for r in rows])  # (M, C)
    labels = [r["label"] for r in rows]

    for c, (cls, col) in enumerate(zip(class_names, colors)):
        ax.barh(
            y,
            shap_mat[:, c],
            left=left,
            height=0.70,
            color=col,
            edgecolor="black",
            linewidth=0.6,
            label=cls
        )
        left += shap_mat[:, c]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (stacked by class)")

    # ---------------------------
    # 4) 组间分隔线（像你图2那样更清晰）
    # ---------------------------
    start = 0
    for g in group_order:
        n = len(groups[g])
        if start > 0:
            ax.axhline(start - 0.5, color="gray", linewidth=0.8, alpha=0.6)
        start += n

    # legend 放图内
    ax.legend(
        frameon=False,
        loc="upper right"
    )

    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig("imgs/1/SHAP_CatLevelsPlusNumeric_StackedByClass_Grouped.png", dpi=600, bbox_inches="tight")
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
