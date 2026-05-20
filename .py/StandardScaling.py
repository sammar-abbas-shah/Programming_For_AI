from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

dataset = pd.DataFrame(
    {
        "Age":[30,20,40,50,33,23,34,60,45,32],
        "SemNo":[1,5,6,2,3,7,8,9,3,2],
        "Attendance":[90,34,89,70,60,50,78,88,99,100]
    }
)

#Method-1 Without SKLearn
#Perform Standardization manually
#Standard Scaling
# Folrmula  Xnew = (X - mean)/SD
# Function to apply Z-score normalization
def zscore_normalizer(df):
    normalized_df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    for column in df.columns:
        mean_val = df[column].mean()  # Calculate mean for the column
        std_dev = df[column].std()  # Calculate standard deviation for the column

        # Apply Z-score normalization formula
        normalized_df[column] = (df[column] - mean_val) / std_dev

    return normalized_df

# Apply Z-score normalization
normalized_df = zscore_normalizer(dataset)

# Display the original and normalized DataFrames
print('=============Without SKLearn================')
print("Original DataFrame:")
print(dataset)
print("\nNormalized DataFrame (Z-score):")
print(normalized_df)
print('=============Without SKLearn================')



#method-2 With SKLearn
print('=============With SK-learn StandScaler================')
sScaler = StandardScaler()
dataset_scaled1 = sScaler.fit_transform(dataset)
dataset_scaled1 = pd.DataFrame(dataset_scaled1,columns=dataset.columns)
print(dataset_scaled1)


#Cross Check - Results of Both
difference = pd.DataFrame()
for feature in dataset.columns:
    difference[feature+'_D'] = np.where(round(normalized_df[feature],4)==round(dataset_scaled1[feature],4),0,1)

print(difference.sum())