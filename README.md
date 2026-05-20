# Programming For AI

A comprehensive Python toolkit covering data preprocessing, feature engineering, and machine learning preparation techniques used in real-world AI and data science workflows.

---

## Overview

This repository contains materials from my **4th Semester — Programming for AI** course. It is a hands-on collection of Python scripts and Jupyter notebooks focused on the **data preprocessing pipeline** — the critical foundation before any machine learning model can be trained. It covers everything from basic DataFrame operations and data visualization, through scaling and encoding, to handling missing values, outliers, class imbalance, and cross-validation.

Whether you're a beginner learning the ropes or a practitioner looking for clean reference implementations, this repo has you covered.

---

## Repository Structure

```
Programming_For_AI/
│
├── .IPYNB/                    # Jupyter Notebook versions
│   ├── balancing.ipynb
│   ├── ColumnTransfer.ipynb
│   ├── DataframeBasic.ipynb
│   ├── FeatureConstructionAndSplitting.ipynb
│   ├── Imbalance_SMOTE_make_classification.ipynb
│   ├── Imbalance_underSampling.ipynb
│   ├── K_Fold_Using_LogisticRegression.ipynb
│   ├── KnnImputation.ipynb
│   ├── MinMaxScaling.ipynb
│   ├── OneHotEncoding.ipynb
│   ├── OutlierDetectionUsingIQR.ipynb
│   ├── OutlierDetectionUsingZScore.ipynb
│   ├── RobustScaling.ipynb
│   ├── StandardScaling.ipynb
│   └── Visualizations.ipynb
│
├── .py/                       # Python script versions
│   ├── balancing.py
│   ├── ColumnTransfer.py
│   ├── DataframeBasic.py
│   ├── FeatureConstructionAndSplitting.py
│   ├── Imbalance_SMOTE_make_classification.py
│   ├── Imbalance_underSampling.py
│   ├── K_Fold_Using_LogisticRegression.py
│   ├── KnnImputation.py
│   ├── MinMaxScaling.py
│   ├── OneHotEncoding.py
│   ├── OutlierDetectionUsingIQR.py
│   ├── OutlierDetectionUsingZScore.py
│   ├── RobustScaling.py
│   ├── StandardScaling.py
│   └── visualizations.py
│
├── csv/                       # Dataset files
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
└── requirements.txt           # Python dependencies
```

---

## Scripts Overview

### DataFrame Operations
| Script | Description |
|--------|-------------|
| `DataframeBasic.py` / `.ipynb` | Creating DataFrames, column manipulation, handling missing values, summary statistics |

### Visualization
| Script | Description |
|--------|-------------|
| `visualizations.py` / `.ipynb` | Data visualization using matplotlib and seaborn for EDA |

### Feature Scaling
| Script | Description |
|--------|-------------|
| `MinMaxScaling.py` / `.ipynb` | Min-Max Normalization to scale features to [0, 1] range |
| `StandardScaling.py` / `.ipynb` | Z-Score Standardization (mean=0, std=1) |
| `RobustScaling.py` / `.ipynb` | Robust Scaling using median and IQR for outlier resistance |

### Encoding & Transformation
| Script | Description |
|--------|-------------|
| `OneHotEncoding.py` / `.ipynb` | Convert categorical variables to binary columns |
| `ColumnTransfer.py` / `.ipynb` | Apply different preprocessing to different columns using ColumnTransformer |

### Feature Engineering
| Script | Description |
|--------|-------------|
| `FeatureConstructionAndSplitting.py` / `.ipynb` | Create new features and split data into train/test sets |

### Outlier Handling
| Script | Description |
|--------|-------------|
| `OutlierDetectionUsingIQR.py` / `.ipynb` | Detect outliers using Interquartile Range method |
| `OutlierDetectionUsingZScore.py` / `.ipynb` | Detect outliers using Z-Score method |

### Imputation
| Script | Description |
|--------|-------------|
| `KnnImputation.py` / `.ipynb` | Handle missing values using K-Nearest Neighbors Imputation |

### Class Imbalance
| Script | Description |
|--------|-------------|
| `balancing.py` / `.ipynb` | Random Over/Under Sampling and NearMiss techniques |
| `Imbalance_SMOTE_make_classification.py` / `.ipynb` | Synthetic Minority Oversampling Technique (SMOTE) |
| `Imbalance_underSampling.py` / `.ipynb` | Under-sampling strategies for imbalanced datasets |

### Model Validation
| Script | Description |
|--------|-------------|
| `K_Fold_Using_LogisticRegression.py` / `.ipynb` | K-Fold Cross-Validation with Logistic Regression |

---

## Datasets

| File | Description |
|------|-------------|
| `50_Startups.csv` | Startup investment data for regression practice |
| `cars.csv` | Car attributes dataset |
| `covid_toy.csv` | Toy COVID dataset for classification |
| `creditcard.zip` | Credit card transactions — imbalanced fraud detection dataset |
| `customer.csv` | Customer attributes for segmentation/classification |
| `data1.csv` | General-purpose sample dataset |
| `onecoldata.csv` | Single-column dataset |
| `titanic_toy.csv` | Titanic survival data for imputation/classification |
| `placement.csv` | Student placement data |
| `train.csv` / `test.csv` | Standard ML train-test split files |
| `predicted.csv` | Predicted values output file |
| `weight-height.csv` | Weight and height data for regression |
| `data_science_job.csv` | Data science job listing data |
| `NearMiss_v*.csv` | Pre-balanced outputs from NearMiss under-sampling |
| `SMOTE_balanced.csv` | Pre-balanced output from SMOTE oversampling |
| `RandomOverSampler_balanced.csv` | Pre-balanced output from Random Over Sampler |
| `RandomUnderSampler_balanced.csv` | Pre-balanced output from Random Under Sampler |

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | DataFrame manipulation and data handling |
| `numpy` | Numerical computations |
| `scikit-learn` | Preprocessing, scaling, encoding, cross-validation |
| `imbalanced-learn` | SMOTE, NearMiss, and other resampling techniques |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical data visualization |

---

## Installation & Setup

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
pip install -r requirements.txt
```

### Run Scripts
```bash
python .py/DataframeBasic.py
python .py/MinMaxScaling.py
python .py/K_Fold_Using_LogisticRegression.py
```

### Open Notebooks
```bash
jupyter notebook .IPYNB/
```

---

## Learning Roadmap

For beginners, follow this suggested order:

1. `DataframeBasic` — Pandas fundamentals
2. `Visualizations` — EDA and data exploration
3. `MinMaxScaling`, `StandardScaling`, `RobustScaling` — Feature scaling
4. `OneHotEncoding` — Categorical encoding
5. `ColumnTransfer` — Pipelines and transformers
6. `FeatureConstructionAndSplitting` — Feature engineering
7. `OutlierDetectionUsingIQR`, `OutlierDetectionUsingZScore` — Outlier handling
8. `KnnImputation` — Missing value imputation
9. `balancing`, `Imbalance_SMOTE_make_classification`, `Imbalance_underSampling` — Class imbalance
10. `K_Fold_Using_LogisticRegression` — Model validation

---

## Contributing

Contributions are welcome! If you'd like to add a new technique, fix a bug, or improve documentation:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Author

**Sammar Abbas Shah**
- GitHub: [@sammar-abbas-shah](https://github.com/sammar-abbas-shah)
- LinkedIn: [sammar-abbas-372359289](https://www.linkedin.com/in/sammar-abbas-372359289/)

---

## License

This project is open-source and available for educational use.