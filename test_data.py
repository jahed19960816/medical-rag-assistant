import pandas as pd

# Load CSV file
df = pd.read_csv("data/medical_abstracts.csv")

# Show first 5 rows
print(df.head())

# Show column names
print(df.columns)