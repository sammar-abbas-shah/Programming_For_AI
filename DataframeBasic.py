import pandas as pd
import numpy as np


# Creating a list of lists
data_list = [
    [1, 'Alice', 25],
    [2, 'Bob', 30],
    [3, 'Charlie', 35],
    [4, 'David', 40],
    [5, 'Eve', 45]
]
# Creating a DataFrame from the list
df_from_list = pd.DataFrame(data_list, columns=['ID', 'Name', 'Age'],index=[11,22,33,44,55])
print(df_from_list)


# Creating a dictionary
data_dict = {
'ID': [1, 2, 3, 4, 5],
'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'], 'Age': [25, 30, 35, 40, 45]
}
# Creating a DataFrame from the dictionary
df_from_dict = pd.DataFrame(data_dict)
print(df_from_dict)

# Create a DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
# Remove column 'C' using pop()
column_c = df.pop('C')
# OR use
del df['A']
# Display the DataFrame after removing column 'C'
print("DataFrame after using pop() to remove column 'C':")
print(df)


# Create a DataFrame with missing values
data = {'A': [1, np.nan, np.nan, 4], 'B': [np.nan, 6, 7, np.nan],
'C': [9, 10, 11, np.nan],'D': [np.nan, 10, 11, np.nan]
        }
df = pd.DataFrame(data)
df.loc[3]={'A':55,'B':np.nan,'C':66,'D':77}
row = df.iloc[0] # as series [:] then Dataframe
print(type(row))
# Display the original DataFrame
print("Original DataFrame:")
print(df)
# Drop rows with at least 2 non-null values
df_cleaned = df.dropna(thresh=3)
# Display the DataFrame after dropping rows with fewer than 2 non-null values
print("\nDataFrame after dropping rows with fewer than 2 non-null values:")
print(df_cleaned)
print(df.describe())