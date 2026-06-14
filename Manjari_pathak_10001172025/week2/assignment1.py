import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("Dataset 2.csv")

# =========================
# Part A: Dataset Understanding
# =========================

print("Q1. First 5 Records")
print(df.head())

print("\nQ2. Shape of Dataset")
print(df.shape)

print("\nQ3. Column Names")
print(df.columns)

print("\nQ4. Data Types")
print(df.dtypes)

print("\nQ5. Missing Values")
print(df.isnull().sum())

# =========================
# Part B: Exploratory Data Analysis
# =========================

print("\nQ6. Average Age")
print(df["Age"].mean())

print("\nQ7. Average Watch Hours Per Week")
print(df["WatchHoursPerWeek"].mean())

print("\nQ8. Average Monthly Spending")
print(df["MonthlySpend"].mean())

print("\nQ9. Subscription Category Counts")
print(df["SubscriptionType"].value_counts())

print("\nQ10. Percentage of Renewed Subscriptions")

renew_percent = (
    (df["SubscriptionRenewed"] == "Yes").sum()
    / len(df)
) * 100

print(renew_percent)

# =========================
# Part C: Data Preparation
# =========================

le = LabelEncoder()

for col in ["Gender",
            "SubscriptionType",
            "FavoriteGenre",
            "SubscriptionRenewed"]:
    df[col] = le.fit_transform(df[col])

print("\nQ11. Encoded Dataset")
print(df.head())

X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]

print("\nQ12. Features and Target Created")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nQ13. Train-Test Split Completed")

# =========================
# Part D: Decision Tree
# =========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nQ14. Decision Tree Trained")

print("\nQ15. Decision Tree Accuracy")
print(dt_accuracy)

print("\nQ16. Confusion Matrix")
print(confusion_matrix(y_test, y_pred_dt))

# =========================
# Part E: KNN
# =========================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nQ17. KNN Trained")

print("\nQ18. KNN Accuracy")
print(knn_accuracy)

print("\nComparison")
print("Decision Tree Accuracy:", dt_accuracy)
print("KNN Accuracy:", knn_accuracy)

# =========================
# Part F: Linear Regression
# =========================

X_reg = df.drop("MonthlySpend", axis=1)
y_reg = df["MonthlySpend"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()

lr.fit(X_train_reg, y_train_reg)

print("\nQ19. Linear Regression Trained")

new_user = [[
    1001, 25, 1, 1, 15, 2, 1, 10, 1
]]

predicted_spend = lr.predict(new_user)

print("\nQ20. Predicted Monthly Spending")
print(predicted_spend[0])
