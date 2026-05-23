# Scrolled to Sadness

Predicting depression risk in teenagers based on social media usage and lifestyle habits using logistic regression.

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
**Source:** Teen Mental Health Dataset

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

## Project Structure

```
scrolled-to-sadness/
├── data/
│   └── Teen_Mental_Health_Dataset.csv
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── outputs/
│   └── figures/                   # Plots generated during EDA
├── .vscode/
│   └── settings.json              # Interpreter settings for VS Code
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/scrolled-to-sadness.git
cd scrolled-to-sadness
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
- [ ] Logistic Regression baseline model
- [ ] Model evaluation (ROC-AUC, F1, confusion matrix)
- [ ] Hyperparameter tuning

---

## Tech Stack

![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=flat-square&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## License

This project is licensed under the MIT License.
