# thermal-runaway-severity-prediction

This repository presents a **physics-informed machine learning framework for predicting the
severity of lithium-ion battery thermal runaway (TR) events using metadata only**, without
requiring calorimetry or mass-ejection measurements as model inputs.

The framework is developed based on the **Battery Failure Databank (v2)** published by the **National Renewable Energy Laboratory (NREL)** and **NASA**, and enables rapid, low-cost, and
scalable assessment of thermal runaway risk prior to event occurrence.


## Problem & Contribution

Thermal runaway (TR) severity is a critical yet under-characterized dimension of battery safety.
Existing experimental characterization methods (e.g., ARC, FTRC) are costly, time-consuming,
and difficult to scale for early-stage battery screening or large datasets.

**This work makes three key contributions:**

1. Proposes a **physics-informed Severity Index (SI)** that integrates normalized heat release
   and mass ejection to quantify the severity of TR events.
2. Develops a **metadata-only machine learning framework** to predict TR severity *prior to event
   occurrence*, eliminating the dependence on calorimetry or post-mortem measurements.
3. Demonstrates **accurate and interpretable severity classification** using supervised models
   (particularly **CatBoost**) trained on the **NREL–NASA Battery Failure Databank**.


## Severity Index (SI)

A physics-informed **Severity Index (SI)** is constructed by combining normalized
heat release and mass ejection metrics reported in the Battery Failure Databank.
The SI serves as a continuous measure of thermal runaway severity, which is further discretized
into **Low / Medium / High** severity levels for supervised classification.

This formulation enables:

- Consistent comparison across heterogeneous battery failure events  
- Severity-aware learning beyond binary TR detection  
- Physics-guided supervision for machine learning models  


## Method Overview

Using the constructed Severity Index, supervised machine-learning models are trained to predict
TR severity based solely on **pre-failure metadata**, including cell specifications and test conditions.

Key features of the framework include:

- Metadata-only inputs (no calorimetry or mass-ejection data required)
- Supervised multiclass classification (Low / Medium / High severity)
- Model interpretability via **SHAP analysis**
- Benchmark comparison across multiple classifiers

Among the evaluated models, **CatBoost** demonstrates strong performance and robustness for
thermal runaway severity prediction.


## Repository Structure

```text
├── data/
│   ├── BatteryFailureDatabankV2.xlsx
│   └── dataframe_processed.xlsx
│
├── imgs/
│   images output by programs
│
├── models/
│   Saved trained models, predictions, and evaluation metrics
│
├── src/
│   ├── analysis.py
│   ├── data.py
│   └── models.py
│  
│    
│
├── 1.main.py
│   End-to-end pipeline: preprocessing, SI construction, model training, and SHAP analysis
│
├── 2.Benchmarks.py
│   Baseline classification models for TR severity comparison
│
├── 3.Catboost_TR_severity_evaluation.py
│   Detailed CatBoost training, error analysis, and class-wise evaluation
│
├── Sensitivity Analysis.ipynb
│   Impact of severity quantile definitions on classification robustness
│
├── plot1.ipynb
├── plot2.ipynb
│   Visualization of data distributions
│
└── environment.yml

```

## Installation

It is recommended to use **conda** to reproduce the environment.

```bash
conda env create -f environment.yml
conda activate thermal-runaway-severity
```



## Usage

Train the thermal runaway severity classification model:

```bash
python 1.main.py
```

The script will:

- Preprocess FTRC battery failure data
- Construct a Severity Index (SI)
- Train a CatBoost multiclass classifier (Low / Medium / High)
- Evaluate performance with cross-validation and test split
- Perform SHAP-based global and class-specific interpretability
- Save trained models, metrics, and publication-quality figures



## Applications

This framework enables:

- Early-stage battery safety screening
- Rapid assessment of thermal runaway risk without calorimetry
- Data-driven comparison of battery designs and chemistries
- Scalable hazard analysis for energy storage systems
- Research on severity-aware battery failure modeling

By eliminating the dependence on calorimetry experiments, the method provides a practical
alternative to traditional ARC or FTRC-based experimental characterization.



## Citation
Manuscript in preparation.
A preprint will be released upon submission.

## Author
Qiqi Hu(胡齐齐)

Institute of Materials Research

Tsinghua Shenzhen International Graduate School

Tsinghua University

Shenzhen 518055 Guangdong

PR China

## Acknowledgements

This work makes use of the **[Battery Failure Databank](https://www.nrel.gov/transportation/battery-failure.html)** provided by the  
**National Renewable Energy Laboratory (NREL)** and **NASA**.