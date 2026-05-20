import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Simulate imbalance
X_minority = X[y == 1].sample(frac=0.2, random_state=42)
y_minority = y[y == 1].sample(frac=0.2, random_state=42)

X_majority = X[y == 0]
y_majority = y[y == 0]

X_imbalanced = pd.concat([X_majority, X_minority], ignore_index=True)
y_imbalanced = pd.concat([y_majority, y_minority], ignore_index=True)

print("Original Class Distribution:\n")
print(y_imbalanced.value_counts())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imbalanced)

# Dictionary of samplers
samplers = {
    "RandomOverSampler": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "RandomUnderSampler": RandomUnderSampler(random_state=42),
    "NearMiss_v1": NearMiss(version=1),
    "NearMiss_v2": NearMiss(version=2),
    "NearMiss_v3": NearMiss(version=3)
}

# Apply each sampler
for name, sampler in samplers.items():

    X_resampled, y_resampled = sampler.fit_resample(
        X_scaled,
        y_imbalanced
    )

    df_resampled = pd.DataFrame(
        X_resampled,
        columns=data.feature_names
    )

    df_resampled['target'] = y_resampled

    # Save to CSV
    filename = f"{name}_balanced.csv"
    df_resampled.to_csv(filename, index=False)
    print(f"\n{name} Class Distribution:\n")
    print(df_resampled['target'].value_counts())

    print(f"Saved as: {filename}")