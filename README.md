# 🤖 Programming For AI

> A comprehensive Python toolkit covering the core data preprocessing, feature engineering, and machine learning preparation techniques used in real-world AI and data science workflows.

---

## 📌 Overview

This repository contains the work done during my **4th Semester — Programming for AI** course. It is a hands-on collection of Python scripts focused on the **data preprocessing pipeline** — the critical foundation before any machine learning model can be trained. It covers everything from basic DataFrame operations and data visualization, through scaling and encoding, to handling missing values, outliers, class imbalance, and cross-validation.

Whether you're a beginner learning the ropes or a practitioner looking for clean reference implementations, this repo has you covered.

---

## 📁 Repository Structure

```
Programming_For_AI/
│
├── 📊 Data Files (CSV / ZIP)
│   ├── 50_Startups.csv
│   ├── cars.csv
│   ├── covid_toy.csv
│   ├── creditcard.zip
│   ├── customer.csv
│   ├── data1.csv
│   ├── onecoldata.csv
│   ├── placement.csv
│   ├── predicted.csv
│   ├── test.csv
│   ├── train.csv
│   ├── titanic_toy.csv
│   ├── weight-height.csv
│   ├── data_science_job.csv
│   ├── NearMiss_v1_balanced.csv
│   ├── NearMiss_v2_balanced.csv
│   ├── NearMiss_v3_balanced.csv
│   ├── RandomOverSampler_balanced.csv
│   ├── RandomUnderSampler_balanced.csv
│   └── SMOTE_balanced.csv
│
└── 🐍 Python Scripts
    ├── DataframeBasic.py
    ├── visualizations.py
    ├── MinMaxScaling.py
    ├── StandardScaling.py
    ├── RobustScaling.py
    ├── OneHotEncoding.py
    ├── ColumnTransfer.py
    ├── FeatureConstructionAndSplitting.py
    ├── OutlierDetectionUsingIQR.py
    ├── OutlierDetectionUsingZScore.py
    ├── KnnImputation.py
    ├── balncing.py
    ├── Imbalance_SMOTE_make_classification.py
    ├── Imbalance_underSampling.py
    └── K_Fold_Using_LogisticRegression.py
```

---

## 🐍 Scripts — Detailed Breakdown

### 🔹 DataframeBasic.py
Covers the essentials of working with **pandas DataFrames**:
- Creating DataFrames from lists and dictionaries with custom indices
- Adding, removing, and accessing columns (`pop()`, `del`, `.iloc[]`, `.loc[]`)
- Handling missing values with `dropna(thresh=…)`
- Summary statistics using `describe()`

**Datasets used:** inline / synthetic data

---

### 🔹 visualizations.py
Demonstrates **data visualization** techniques using `matplotlib` and/or `seaborn`:
- Plotting distributions, relationships, and patterns in datasets
- Useful for exploratory data analysis (EDA) before modeling

**Datasets used:** various CSVs in the repo

---

### 🔹 MinMaxScaling.py
Implements **Min-Max Normalization** using `sklearn.preprocessing.MinMaxScaler`:
- Scales features to a fixed range (typically [0, 1])
- Ideal when the algorithm requires bounded input (e.g., neural networks, KNN)

---

### 🔹 StandardScaling.py
Implements **Z-Score Standardization** using `sklearn.preprocessing.StandardScaler`:
- Centers data around mean = 0 and standard deviation = 1
- Best suited for algorithms that assume normally distributed features (e.g., linear regression, SVM)

---

### 🔹 RobustScaling.py
Implements **Robust Scaling** using `sklearn.preprocessing.RobustScaler`:
- Uses median and IQR instead of mean/std — making it resistant to outliers
- Preferred when data contains significant outliers

---

### 🔹 OneHotEncoding.py
Demonstrates **One-Hot Encoding** for converting categorical variables into binary columns:
- Uses `pd.get_dummies()` or `sklearn.preprocessing.OneHotEncoder`
- Essential for algorithms that cannot handle string/category inputs

---

### 🔹 ColumnTransfer.py
Shows how to apply **ColumnTransformer** from `sklearn.compose`:
- Applies different preprocessing steps (scaling, encoding) to different columns in a single pipeline
- Keeps preprocessing clean and production-ready

---

### 🔹 FeatureConstructionAndSplitting.py
Covers two fundamental ML preparation steps:
- **Feature Construction**: Creating new features from existing ones to improve model performance
- **Train/Test Split**: Splitting data using `sklearn.model_selection.train_test_split`

---

### 🔹 OutlierDetectionUsingIQR.py
Detects and handles outliers using the **Interquartile Range (IQR)** method:
- Computes Q1, Q3, and IQR
- Flags or removes rows where values fall outside `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`

---

### 🔹 OutlierDetectionUsingZScore.py
Detects outliers using the **Z-Score** method:
- Marks data points with |z| > 3 as outliers
- Useful when data is approximately normally distributed

---

### 🔹 KnnImputation.py
Handles missing values using **K-Nearest Neighbors Imputation** (`sklearn.impute.KNNImputer`):
- More sophisticated than mean/median imputation — uses neighboring data points to fill gaps
- Works well when missing data has patterns related to other features

**Datasets used:** `titanic_toy.csv`, `data_science_job.csv`

---

### 🔹 balncing.py
Demonstrates multiple **class balancing** strategies for imbalanced datasets:
- Random Over Sampling
- Random Under Sampling
- NearMiss v1, v2, v3 (proximity-based under-sampling)
- Outputs balanced CSVs: `RandomOverSampler_balanced.csv`, `RandomUnderSampler_balanced.csv`, `NearMiss_v*.csv`

**Datasets used:** `creditcard.zip`

---

### 🔹 Imbalance_SMOTE_make_classification.py
Applies **SMOTE (Synthetic Minority Oversampling Technique)** to synthetically generate minority class samples:
- Uses `imblearn.over_sampling.SMOTE`
- Generates synthetic data via `sklearn.datasets.make_classification`
- Outputs `SMOTE_balanced.csv`

---

### 🔹 Imbalance_underSampling.py
Focuses specifically on **under-sampling** techniques for imbalanced datasets:
- Implements NearMiss strategies and random under-sampling
- Compares different under-sampling approaches on a classification dataset

---

### 🔹 K_Fold_Using_LogisticRegression.py
Demonstrates **K-Fold Cross-Validation** with Logistic Regression:
- Uses `sklearn.model_selection.KFold` / `cross_val_score`
- Evaluates model performance more reliably than a single train/test split
- Reduces variance in performance estimation

---

## 📦 Datasets

| File | Description |
|---|---|
| `50_Startups.csv` | Startup investment data for regression practice |
| `cars.csv` | Car attributes dataset |
| `covid_toy.csv` | Toy COVID dataset for classification |
| `creditcard.zip` | Credit card transactions — imbalanced fraud detection dataset |
| `customer.csv` | Customer attributes for segmentation/classification |
| `data1.csv` | General-purpose sample dataset |
| `titanic_toy.csv` | Titanic survival data for imputation/classification |
| `placement.csv` | Student placement data |
| `train.csv` / `test.csv` | Standard ML train-test split files |
| `weight-height.csv` | Weight and height data for regression |
| `data_science_job.csv` | Data science job listing data |
| `NearMiss_v*.csv` | Pre-balanced outputs from NearMiss under-sampling |
| `SMOTE_balanced.csv` | Pre-balanced output from SMOTE oversampling |
| `RandomOverSampler_balanced.csv` | Pre-balanced output from Random Over Sampler |
| `RandomUnderSampler_balanced.csv` | Pre-balanced output from Random Under Sampler |

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | DataFrame manipulation and data handling |
| `numpy` | Numerical computations |
| `scikit-learn` | Preprocessing, scaling, encoding, cross-validation |
| `imbalanced-learn` | SMOTE, NearMiss, and other resampling techniques |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical data visualization |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Clone the Repository
```bash
git clone https://github.com/sammar-abbas-shah/Programming_For_AI.git
cd Programming_For_AI
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Run Any Script
```bash
python DataframeBasic.py
python MinMaxScaling.py
python K_Fold_Using_LogisticRegression.py
# ... and so on
```

---

## 🗺️ Learning Roadmap

If you're using this repo to learn, here's a suggested order:

```
1. DataframeBasic.py          → Pandas fundamentals
2. visualizations.py          → EDA & plotting
3. MinMaxScaling.py           → Feature scaling
   StandardScaling.py
   RobustScaling.py
4. OneHotEncoding.py          → Categorical encoding
5. ColumnTransfer.py          → Pipelines & transformers
6. FeatureConstructionAndSplitting.py → Feature engineering
7. OutlierDetectionUsingIQR.py        → Outlier handling
   OutlierDetectionUsingZScore.py
8. KnnImputation.py           → Missing value imputation
9. balncing.py                → Class imbalance
   Imbalance_SMOTE_make_classification.py
   Imbalance_underSampling.py
10. K_Fold_Using_LogisticRegression.py → Model validation
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to add a new technique, fix a bug, or improve documentation:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👩‍💻 Author

**Sammar Abbas Shah**
- GitHub: [@sammar-abbas-shah](https://github.com/sammar-abbas-shah)
- LinkedIn: [sammar-abbas-372359289](https://www.linkedin.com/in/sammar-abbas-372359289/)

---

## 📄 License

This project is open-source and available for educational use.

---

> *"Before you can train a model, you must first understand your data."*
