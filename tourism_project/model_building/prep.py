import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Loaded dataset for preparation.")

# Data cleaning: Standardize Gender values
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Drop unique identifier and index columns
cols_to_drop = [c for c in df.columns if c in ['CustomerID'] or c.startswith('Unnamed') or c == '']
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Impute missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

target_col = 'ProdTaken'
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train-test splits saved locally as CSV files.")
