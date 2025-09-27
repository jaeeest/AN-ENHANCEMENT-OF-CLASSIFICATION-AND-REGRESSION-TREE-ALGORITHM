import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib
import time

# -------------------------
# Weighted ADASYN function
# -------------------------
def weighted_euclidean_distance(p, q, weights):
    return np.sqrt(np.sum(weights * (p - q) ** 2))

def weighted_adasyn(X, y, beta=1.0, k=5, weights=None):
    from sklearn.neighbors import NearestNeighbors
    X_min = X[y == 1]
    X_maj = X[y == 0]
    n_min, n_maj = len(X_min), len(X_maj)
    G = int((n_maj - n_min) * beta)
    if weights is None:
        weights = np.ones(X.shape[1])
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    synthetic = []
    for xi in X_min:
        distances, indices = nn.kneighbors([xi])
        for idx in indices[0]:
            if y[idx] == 0:  
                continue
            xj = X[idx]
            diff = xj - xi
            gap = np.random.rand()
            new_point = xi + gap * diff
            synthetic.append(new_point)
            if len(synthetic) >= G:
                break
        if len(synthetic) >= G:
            break
    X_syn = np.array(synthetic)
    y_syn = np.ones(len(X_syn))
    return np.vstack([X, X_syn]), np.hstack([y, y_syn])

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("onlinefraud.csv")

features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

X = df[features].values
y = df["isFraud"].values

feature_weights = np.ones(X.shape[1])

print("Before ADASYN:", np.bincount(y))
X_resampled, y_resampled = weighted_adasyn(X, y, beta=0.8, k=5, weights=feature_weights)
print("After ADASYN :", np.bincount(y_resampled.astype(int)))

# -------------------------
# Train CART + AdaBoost with Regularized Gini
# -------------------------
cart = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=0.001,         # regularization
    random_state=42
)

ada_cart = AdaBoostClassifier(
    estimator=cart,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# Timing training
print("Starting AdaBoost training...")
start_time = time.time()
ada_cart.fit(X_resampled, y_resampled.astype(int))
end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds")

# Timing saving
print("💾 Saving model...")
save_start = time.time()
joblib.dump({"model": ada_cart, "features": features}, "enhanced_cart.pkl", compress=0)
save_end = time.time()
print(f"Model and features saved as enhanced_cart.pkl (took {save_end - save_start:.2f} seconds)")
