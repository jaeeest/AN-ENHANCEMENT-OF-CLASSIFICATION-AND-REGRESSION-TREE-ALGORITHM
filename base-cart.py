import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv("onlinefraud.csv")

# Encode categorical features
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop("isFraud", axis=1).values
y = df["isFraud"].values

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ============================================================
# Base CART (Original)
# ============================================================
cart = DecisionTreeClassifier(
    criterion="gini",   # CART uses Gini by default
    random_state=42
)

cart.fit(X_train, y_train)
y_pred = cart.predict(X_test)

# ============================================================
# Evaluation
# ============================================================
print("\n--- Base CART Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
