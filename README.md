# thermal-runaway-severity-prediction

Leveraging the [Battery Failure Databank](https://www.nrel.gov/transportation/battery-failure.html) published by NREL and NASA, this repository provides a machine-learning framework for **predicting the severity of lithium-ion battery thermal runaway events using metadata only**, without requiring calorimetry or mass-ejection measurements as model inputs. A physics-informed **Severity Index (SI)** is constructed by combining normalized heat release and mass ejection metrics, and batteries are categorized into **Low / Medium / High severity levels**.  
Using this formulation, supervised machine-learning models—particularly **CatBoost**—are trained to predict thermal runaway severity prior to event occurrence.

This approach enables **rapid, low-cost, and scalable assessment of thermal runaway risk**, providing a practical alternative to traditional ARC or FTRC-based experimental characterization.



## Repository structure

- `data` folder contains a copy of the battery failure databank (version 2), and the processed dataframe in this research
- `src` folder contains classes and methods for data processing, model definitions and training, and analysis tools for plotting results
- `1.main.py` trains models and conducts SHAP interpretability
- `2.Benchmarks.py` compares several benchmarking classification algorithms for TR severity
- `3.Catboost_TR_severity_evaluation.py` implements catBoost classification and error analysis for TR Severity
- `Sensitivity Analysis.ipynb` compares impact of different severity quantile schemes on catBoost classification
- `models` folder saves trained model and predictions and errors
- `plot1.ipynb` and 'plot2.ipynb' creates plots of the data and results, as shown in the manuscript



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

This framework is intended for:

- Battery safety screening  
- Early-stage cell risk assessment  
- Energy storage system hazard analysis  
- Data-driven battery safety research  
- Engineering-level comparison of battery designs  

By eliminating the dependence on calorimetry experiments, the method enables faster and more accessible thermal runaway risk evaluation.



## Citation

Coming soon...



## Author

Coming soon...

---

## Acknowledgements

This work makes use of the **Battery Failure Databank** provided by the  
**National Renewable Energy Laboratory (NREL)** and **NASA**.
