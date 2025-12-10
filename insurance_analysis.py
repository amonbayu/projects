
import pandas as pd
#load the dataset checking for missing values and the data types
df = pd.read_csv('insurance.csv')
print(df.columns)
df.isnull().sum()
df.dtypes
num_cols = ['age', 'bmi', 'children', 'charges']
# Remove outliers based on 1st and 99th percentiles
for col in num_cols:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df = df[(df[col] >= low) & (df[col] <= high)]
    print(col, low, high)
# Encode categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
df_encoded = df_encoded.apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

#show results
print(df_encoded.head())
print(df_encoded.dtypes)
#saving the csv file
df_encoded.to_csv('insurance_cleaned.csv', index=False)
