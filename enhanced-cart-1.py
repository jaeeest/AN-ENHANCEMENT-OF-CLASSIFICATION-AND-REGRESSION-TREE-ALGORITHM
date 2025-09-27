import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Weighted Euclidean Distance
# ============================================================
def weighted_euclidean_distance(p, q, weights):
    return np.sqrt(np.sum(weights * (p - q) ** 2))

# ============================================================
# Weighted ADASYN (simplified for demonstration)
# ============================================================
from sklearn.neighbors import NearestNeighbors

def weighted_adasyn(X, y, beta=1.0, k=5, weights=None):
    X_min = X[y == 1]
    X_maj = X[y == 0]

    n_min, n_maj = len(X_min), len(X_maj)
    G = (n_maj - n_min) * beta  # number of synthetic samples

    synthetic_samples = []
    y_synthetic = []

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    for xi in X_min:
        distances, indices = nn.kneighbors([xi])
        for idx in indices[0]:
            if y[idx] == 0:
                continue
            xj = X[idx]
            lam = np.random.rand()
            diff = (xj - xi)
            if weights is not None:
                diff = diff * np.sqrt(weights)
            synthetic = xi + lam * diff
            synthetic_samples.append(synthetic)
            y_synthetic.append(1)
            if len(synthetic_samples) >= G:
                break
        if len(synthetic_samples) >= G:
            break

    if len(synthetic_samples) > 0:
        X_new = np.vstack([X, synthetic_samples])
        y_new = np.hstack([y, y_synthetic])
    else:
        X_new, y_new = X, y

    return X_new, y_new

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv("onlinefraud.csv")

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#df = df.sample(100000, random_state=42) #smaller sample size

# Split features and target
X = df.drop("isFraud", axis=1).values
y = df["isFraud"].values

# ============================================================
# Weighted ADASYN
# ============================================================
feature_weights = np.ones(X.shape[1]) 
X_resampled, y_resampled = weighted_adasyn(X, y, beta=0.8, k=5, weights=feature_weights)

print("Before ADASYN:", np.bincount(y))
print("After ADASYN :", np.bincount(y_resampled))

# ============================================================
# Train/Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
)

# ============================================================
# CART + AdaBoost + Regularized Gini
# ============================================================
cart = DecisionTreeClassifier(
    criterion="gini",       # standard Gini (approximating Regularized Gini)
    max_depth=5,            # limits tree growth (stability)
    min_samples_split=10,   
    min_samples_leaf=5,    
    ccp_alpha=0.001,        # cost-complexity pruning (acts like λ·Complexity(T))
    random_state=42
)

ada_cart = AdaBoostClassifier(
    estimator=cart,
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

ada_cart.fit(X_train, y_train)
y_pred = ada_cart.predict(X_test)

# ============================================================
# Evaluation
# ============================================================
print("\n--- Final Enhanced CART Results (with Regularized Gini Approximation) ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

