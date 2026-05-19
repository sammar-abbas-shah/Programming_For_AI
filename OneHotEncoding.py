import pandas as pd
import numpy as np

df = pd.read_csv('cars.csv')
print(df.sample(5))
print(df.shape)
print("======Unique values in each column======")
unique_brands = df.nunique()
print(unique_brands)

#USING PANDAS
#apply OHE using pandas on fuel and owner columns
ohe_encoded = pd.get_dummies(df,columns=['fuel','owner'])
print(ohe_encoded.columns)

#to handle multi-collinearity drop 1st from both
ohe_encoded1 = pd.get_dummies(df,columns=['fuel','owner'],drop_first=True)
print(ohe_encoded1.columns)
print(ohe_encoded1)


#USING SKLEARN
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first')#, sparse=False
#sparse=False : then no need ot .toarray()
#a numpy type array
ohe_encoded2 = ohe.fit_transform(df[['fuel','owner']]).toarray()
print(ohe_encoded2)

overall_data = np.hstack((df[['brand','km_driven']].values,ohe_encoded2))
print(overall_data.shape)

