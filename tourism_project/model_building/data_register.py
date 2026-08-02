import pandas as pd
import os

DATASET_PATH = "tourism_project/data/tourism.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", 
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting", 
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", 
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore", 
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

actual_columns = list(df.columns)
missing_columns = [col for col in expected_columns if col not in actual_columns]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")
else:
    print("All expected columns are present.")

print(f"Dataset Shape: {df.shape}")
print("Missing values summary:")
print(df.isnull().sum())
print("Target distribution ('ProdTaken'):")
print(df['ProdTaken'].value_counts(normalize=True))
