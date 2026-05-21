# Programming For AI Repository

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

> A comprehensive Python toolkit covering data preprocessing, feature engineering, and machine learning preparation techniques used in real-world AI and data science workflows.

## 📖 Overview

Materials from **4th Semester — Programming for AI course**. This hands-on collection is focused on the complete data preprocessing pipeline, covering DataFrame operations, data visualization, feature scaling, encoding, handling missing values, outliers, class imbalance, and cross-validation techniques essential for building robust ML models.

## 📁 Repository Structure

```
Programming_For_AI/
├── .IPYNB/                # Jupyter Notebooks (interactive learning)
│   ├── DataframeBasic.ipynb
│   ├── Visualizations.ipynb
│   ├── MinMaxScaling.ipynb
│   ├── StandardScaling.ipynb
│   ├── RobustScaling.ipynb
│   ├── OneHotEncoding.ipynb
│   ├── ColumnTransfer.ipynb
│   ├── FeatureConstructionAndSplitting.ipynb
│   ├── OutlierDetectionUsingIQR.ipynb
│   ├── OutlierDetectionUsingZScore.ipynb
│   ├── KnnImputation.ipynb
│   ├── balancing.ipynb
│   ├── Imbalance_SMOTE_make_classification.ipynb
│   ├── Imbalance_underSampling.ipynb
│   └── K_Fold_Using_LogisticRegression.ipynb
├── csv/                    # Sample datasets for practice
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Required packages (listed in requirements.txt)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sammar-abbas-shah/Programming_For_AI.git
cd Programming_For_AI
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook
```

## 📚 Contents

### DataFrame Operations
| File | Description |
|------|-------------|
| `DataframeBasic` | Creating DataFrames, column manipulation, handling missing values, summary statistics |

### Visualization
| File | Description |
|------|-------------|
| `Visualizations` | Data visualization using matplotlib and seaborn for exploratory data analysis (EDA) |

### Feature Scaling
| File | Description |
|------|-------------|
| `MinMaxScaling` | Min-Max Normalization to scale features to [0, 1] range |
| `StandardScaling` | Z-Score Standardization (mean=0, std=1) |
| `RobustScaling` | Robust Scaling using median and IQR for outlier resistance |

### Encoding & Transformation
| File | Description |
|------|-------------|
| `OneHotEncoding` | Convert categorical variables to binary columns |
| `ColumnTransfer` | Apply different preprocessing to different columns using ColumnTransformer |

### Feature Engineering
| File | Description |
|------|-------------|
| `FeatureConstructionAndSplitting` | Create new features and split data into train/test sets |

### Outlier Handling
| File | Description |
|------|-------------|
| `OutlierDetectionUsingIQR` | Detect outliers using Interquartile Range method |
| `OutlierDetectionUsingZScore` | Detect outliers using Z-Score method |

### Imputation
| File | Description |
|------|-------------|
| `KnnImputation` | Handle missing values using K-Nearest Neighbors Imputation |

### Class Imbalance
| File | Description |
|------|-------------|
| `balancing` | Random Over/Under Sampling and NearMiss techniques |
| `Imbalance_SMOTE_make_classification` | Synthetic Minority Oversampling Technique (SMOTE) |
| `Imbalance_underSampling` | Under-sampling strategies for imbalanced datasets |

### Model Validation
| File | Description |
|------|-------------|
| `K_Fold_Using_LogisticRegression` | K-Fold Cross-Validation with Logistic Regression |

## 📂 Included Datasets

The `csv/` folder contains sample datasets for practice:

| Dataset | Description |
|---------|-------------|
| `50_Startups.csv` | Startup investment data for regression practice |
| `cars.csv` | Car attributes dataset |
| `covid_toy.csv` | Toy COVID dataset for classification |
| `creditcard.zip` | Credit card transactions — imbalanced fraud detection dataset |
| `titanic_toy.csv` | Titanic survival data for imputation/classification |
| `placement.csv` | Student placement data |
| `weight-height.csv` | Weight and height data for regression |
| `data_science_job.csv` | Data science job listing data |
| `*_balanced.csv` | Pre-balanced outputs from various sampling techniques |

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| pandas | DataFrame manipulation and data handling |
| NumPy | Numerical computations |
| Scikit-learn | Preprocessing, scaling, encoding, cross-validation |
| imbalanced-learn | SMOTE, NearMiss, and other resampling techniques |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |

## 🎓 Learning Roadmap

Recommended sequence for learning:

1. **DataframeBasic** — Pandas fundamentals and data manipulation
2. **Visualizations** — Exploratory Data Analysis (EDA) and data exploration
3. **MinMaxScaling / StandardScaling / RobustScaling** — Feature scaling techniques
4. **OneHotEncoding** — Categorical encoding methods
5. **ColumnTransfer** — Building preprocessing pipelines with ColumnTransformer
6. **FeatureConstructionAndSplitting** — Feature engineering and data splitting
7. **OutlierDetectionUsingIQR / OutlierDetectionUsingZScore** — Outlier detection and handling
8. **KnnImputation** — Advanced missing value imputation
9. **balancing / Imbalance_SMOTE_make_classification / Imbalance_underSampling** — Handling class imbalance
10. **K_Fold_Using_LogisticRegression** — Model validation and cross-validation

## 📧 Contact

For any questions, collaborations, or feedback:

- **Email**: [sammarabbas0619@gmail.com](mailto:sammarabbas0619@gmail.com)
- **LinkedIn**: [Sammar Abbas](https://www.linkedin.com/in/sammar-abbas-372359289/)

## 📝 License

Open-source and available for educational use. Please provide attribution if using any of the code or concepts.

---

⭐ If you found this helpful, please give this repository a star!

*Last updated: 2026*
