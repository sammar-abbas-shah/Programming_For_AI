from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset
df = pd.read_csv('covid_toy.csv')
print(df.head())

# Check for missing values
print(df.isnull().sum())

tlist = ['tnf1', OrdinalEncoder()]

# Apply transformations to the appropriate columns
transformer = ColumnTransformer(transformers=[
   # ('tnf1', SimpleImputer(), ['fever']),  # Impute missing values in 'fever'
    ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
    ('tnf3', OneHotEncoder(sparse_output=False, drop='first'), ['gender', 'city']),
], remainder='passthrough')

# Transform the features
data_encoded = transformer.fit_transform(df)
print(type(data_encoded))

# Print the shape of the transformed data
print(data_encoded.shape)

# If 'has_covid' is the target column, apply LabelEncoder separately
label_encoder = LabelEncoder()

# df['has_covid'] = label_encoder.fit_transform(data_encoded['has_covid'])
# data_encoded[7] = label_encoder.fit_transform(data_encoded[7])

# Display the encoded target column
# print(data_encoded)#['has_covid'].head())
