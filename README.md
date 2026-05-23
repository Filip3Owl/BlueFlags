# BlueFlags

Predicting depression risk in teenagers based on social media usage and lifestyle habits using logistic regression.

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## Overview

This project builds a binary classification pipeline to determine whether a teenager is at risk of depression (`depression_label`: 0 or 1). The dataset captures behavioral and lifestyle patterns — including daily social media consumption, sleep duration, academic performance, and self-reported stress levels — collected from approximately 1,200 adolescents.

The project follows a structured machine learning workflow: exploratory data analysis, feature engineering, preprocessing, model training, and evaluation.

---

## Problem Statement

Adolescent mental health has become a growing concern, with social media usage often cited as a contributing factor to rising rates of anxiety and depression. This project investigates whether behavioral and lifestyle variables can reliably predict depressive symptoms, using logistic regression as the primary model.

**Target variable:** `depression_label` — binary (0 = no depression, 1 = depression)

---

## Dataset

**File:** `data/Teen_Mental_Health_Dataset.csv`  
**Records:** 1,200

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the teenager (years) |
| `gender` | Categorical | Gender (male / female) |
| `daily_social_media_hours` | Numeric | Average hours per day spent on social media |
| `platform_usage` | Categorical | Primary social media platform used |
| `sleep_hours` | Numeric | Average hours of sleep per night |
| `screen_time_before_sleep` | Numeric | Hours of screen exposure before sleeping |
| `academic_performance` | Numeric | GPA-style academic score |
| `physical_activity` | Numeric | Hours of physical activity per week |
| `social_interaction_level` | Categorical | Level of in-person social interaction (low / medium / high) |
| `stress_level` | Numeric | Self-reported stress score (1–10) |
| `anxiety_level` | Numeric | Self-reported anxiety score (1–10) |
| `addiction_level` | Numeric | Social media addiction score (1–10) |
| `depression_label` | Binary | Target variable — 0: no depression, 1: depression |

---

## Exploratory Data Analysis

### Class Distribution

The dataset presents a severe **class imbalance**: 97.4% of records belong to class 0 (no depression) and only 2.6% to class 1 (depression). This imbalance is a critical constraint for modelling and must be addressed through resampling techniques or class-weight adjustment before training.

![Class Distribution](assets/01_target_distribution.png)

---

### Correlation with the Target Variable

The heatmap below shows pairwise Pearson correlations among all numerical features. The last row highlights the relationship of each feature with `depression_label`.

![Correlation Matrix](assets/06_correlation_matrix.png)

The four strongest correlates with depression are:

| Feature | Correlation | Direction |
|---|---|---|
| `sleep_hours` | -0.191 | Less sleep is associated with higher depression risk |
| `daily_social_media_hours` | +0.175 | More time on social media correlates with depression |
| `stress_level` | +0.171 | Higher stress scores track closely with depression |
| `anxiety_level` | +0.170 | Higher anxiety scores track closely with depression |

---

### Feature Distributions by Class

Box plots reveal meaningful shifts in feature distributions between depressed and non-depressed teenagers. The most pronounced differences are observed in `stress_level`, `anxiety_level`, `sleep_hours`, and `daily_social_media_hours`.

![Bivariate Box Plots](assets/04_bivariate_boxplots.png)

**Mean values by class:**

| Feature | No Depression (0) | Depression (1) | Difference |
|---|---|---|---|
| `stress_level` | 5.37 | 8.48 | +3.11 |
| `anxiety_level` | 5.56 | 8.61 | +3.05 |
| `daily_social_media_hours` | 4.48 h | 6.72 h | +2.24 h |
| `sleep_hours` | 6.49 h | 4.76 h | -1.73 h |

---

### Scatter Matrix — Top Features

The scatter matrix below plots pairwise relationships among the four most correlated features, coloured by class. The overlap between classes is consistent with the weak-to-moderate correlation values observed, confirming that no single feature is sufficient for classification on its own.

![Scatter Matrix](assets/07_scatter_matrix.png)

---

## Key Findings

- **Severe class imbalance (97.4% vs 2.6%)** makes accuracy a misleading metric. ROC-AUC and F1-score will be the primary evaluation criteria.
- **Stress and anxiety levels** show the largest mean difference between classes (+3 points on a 1–10 scale), making them the most discriminative features.
- **Depressed teenagers sleep on average 1.73 hours less** per night than non-depressed peers.
- **Social media consumption is 50% higher** in the depressed group (6.72 h vs 4.48 h per day).
- **Academic performance, physical activity, and addiction level** show negligible differences between classes and are unlikely to contribute meaningfully to the model.

---

## Project Structure

```
BlueFlags/
├── assets/                        # Figures embedded in this README
├── data/
│   └── Teen_Mental_Health_Dataset.csv
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── outputs/
│   └── figures/                   # All plots generated during EDA
├── .vscode/
│   └── settings.json
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Filip3Owl/BlueFlags.git
cd BlueFlags
```

### 2. Create and activate the virtual environment

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=venv-logistic --display-name "Python (.venv)"
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Open `notebooks/eda.ipynb` and select the **Python (.venv)** kernel.

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/eda.ipynb` | Exploratory Data Analysis — class balance, univariate and bivariate distributions, correlation analysis |

---

## Roadmap

- [x] Exploratory Data Analysis
- [ ] Feature engineering and preprocessing
- [ ] Handle class imbalance (SMOTE / class\_weight)
- [ ] Logistic Regression baseline model
- [ ] Model evaluation (ROC-AUC, F1, confusion matrix)
- [ ] Hyperparameter tuning

---

## License

This project is licensed under the MIT License.
