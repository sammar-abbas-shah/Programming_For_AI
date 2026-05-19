
import numpy as np
import pandas as pd

dataset = pd.DataFrame(
    {
        "Age":[30,20,40,50,33,23,34,60,45,32],
        "SemNo":[1,5,6,2,3,7,8,9,3,2],
        "Attendance":[90,34,89,70,60,50,78,88,99,100]
    }
)
#MIn-Max Scaler / Normalization

#Method-1 - Without SKLearn
#Perform Normalization manually
#Min-Max Scaling
#Folrmula  NewVal = (CurVal - Xmin)/(Xmax - Xmin)


# Function to apply MinMax scaling
def minmax_scaler(df):
    scaled_df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    for column in df.columns:
        min_val = df[column].min()
        max_val = df[column].max()

        # Apply MinMax scaling formula
        scaled_df[column] = (df[column] - min_val) / (max_val - min_val)

    return scaled_df

# Apply MinMax scaling
scaled_df = minmax_scaler(dataset)

# Display the original and scaled DataFrames
print('=============Without SKLearn================')
print("Original DataFrame:")
print(dataset)
print("\nScaled DataFrame:")
print(scaled_df)
print('=============Without SKLearn================')



#method-2 with SKLearn
print('=============With SK-learn MinMax================')
from sklearn.preprocessing import MinMaxScaler
mmScaler = MinMaxScaler()
dataset_scaled1 = mmScaler.fit_transform(dataset)
dataset_scaled1 = pd.DataFrame(dataset_scaled1,columns=dataset.columns)
print(dataset_scaled1)

# #Cross Check
# difference = pd.DataFrame()
# for feature in dataset.columns:
#     difference[feature+'_D'] = np.where(round(dataset_scaled[feature],4)==round(dataset_scaled1[feature],4),0,1)
#
# print(difference.sum())

import pandas as pd

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
}



