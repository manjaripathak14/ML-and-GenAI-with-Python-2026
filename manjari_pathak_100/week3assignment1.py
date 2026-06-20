# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================
# Q1 Dataset Overview
# ==========================

df = pd.read_csv("agriculture_yield_dataset.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 10 Records:")
print(df.head(10))

# ==========================
# Q2 Data Types & Missing Values
# ==========================

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# Q3 Descriptive Statistics
# ==========================

print("\nSummary Statistics:")
print(df.describe())

# ==========================
# Q4 Histograms
# ==========================

cols = [
    "rainfall_mm",
    "temperature_c",
    "fertilizer_kg",
    "yield_ton_per_hectare"
]

for col in cols:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# ==========================
# Q5 Crop Type Analysis
# ==========================

print("\nCrop Type Counts:")
print(df["crop_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="crop_type", data=df)
plt.title("Crop Type Distribution")
plt.show()

# ==========================
# Q6 Soil Type Analysis
# ==========================

print("\nSoil Type Counts:")
print(df["soil_type"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="soil_type", data=df)
plt.title("Soil Type Distribution")
plt.show()

# ==========================
# Q7 Yield Distribution
# ==========================

plt.figure(figsize=(6,4))
plt.hist(df["yield_ton_per_hectare"], bins=20)
plt.title("Yield Distribution")
plt.xlabel("Yield")
plt.ylabel("Frequency")
plt.show()

# ==========================
# Q8 Scatter Plots
# ==========================

plt.figure(figsize=(6,4))
plt.scatter(
    df["rainfall_mm"],
    df["yield_ton_per_hectare"]
)
plt.xlabel("Rainfall")
plt.ylabel("Yield")
plt.title("Rainfall vs Yield")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(
    df["fertilizer_kg"],
    df["yield_ton_per_hectare"]
)
plt.xlabel("Fertilizer")
plt.ylabel("Yield")
plt.title("Fertilizer vs Yield")
plt.show()

# ==========================
# Q9 Correlation Analysis
# ==========================

corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================
# Q10 Group-Based Analysis
# ==========================

print("\nAverage Yield by Crop Type:")
print(
    df.groupby("crop_type")[
        "yield_ton_per_hectare"
    ].mean()
)

print("\nAverage Yield by Soil Type:")
print(
    df.groupby("soil_type")[
        "yield_ton_per_hectare"
    ].mean()
)

# ==========================
# Q11 One-Hot Encoding
# ==========================

df_encoded = pd.get_dummies(
    df,
    columns=["crop_type", "soil_type"],
    drop_first=True
)

print("\nEncoded Dataset:")
print(df_encoded.head())

# ==========================
# Q12 Feature Selection
# ==========================

X = df_encoded.drop(
    "yield_ton_per_hectare",
    axis=1
)

y = df_encoded[
    "yield_ton_per_hectare"
]

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)

# ==========================
# Q13 Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# ==========================
# Q14 Linear Regression
# ==========================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nIntercept:")
print(model.intercept_)

coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature Coefficients:")
print(
    coef_df.sort_values(
        by="Coefficient",
        ascending=False
    )
)
