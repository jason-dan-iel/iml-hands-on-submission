# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import random

# Reading the data from a CSV file into a pandas DataFrame
df = pd.read_csv("pca-demo-food-country.csv")

# Analysing the Data
df.head()
df.info()
# Dropping the first column from the DataFrame
df.drop(df.columns[0], axis=1, inplace=True)
df

# Initializing a PCA object with 2 components
pca = PCA(n_components=2)
# Fitting the PCA model to the data
pca.fit(df)
# Transforming the original data to the new coordinate system defined by the principal components
new_data = pca.transform(df)

# Creating & Displaying a new DataFrame to store the transformed data
new_data_pca = pd.DataFrame(new_data, columns=["PC1", "PC2"])
new_data_pca
