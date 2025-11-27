import numpy as np
import pandas as pd
import time
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# -------------------------
# Weighted ADASYN function
# -------------------------

def weighted_euclidean_distance(p, q, weights):
    return np.sqrt(np.sum(weights * (p - q) ** 2))

def weighted_adasyn(X, y, beta=1.0, k=5, weights=None):
    X_min = X[y == 1]
    X_maj = X[y == 0]

    n_min, n_maj = len(X_min), len(X_maj)
    G = int((n_maj - n_min) * beta) 

    synthetic_samples = []
    y_synthetic = []

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    
    for xi in X_min:
        if len(synthetic_samples) >= G:
            break
            
        distances, indices = nn.kneighbors([xi])
        
        for idx in indices[0]:
            if len(synthetic_samples) >= G:
                break
                
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


    if len(synthetic_samples) > 0:
        X_new = np.vstack([X, synthetic_samples])
        y_new = np.hstack([y, y_synthetic])
    else:
        X_new, y_new = X, y

    return X_new, y_new

# -------------------------
# Load dataset and Preprocessing
# -------------------------
df = pd.read_csv("onlinefraud.csv")

features = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

X = df[features].values
y = df["isFraud"].values

# -------------------------
# Weighted ADASYN execution
# -------------------------
feature_weights = np.ones(X.shape[1])
print("Before ADASYN:", np.bincount(y))
X_resampled, y_resampled = weighted_adasyn(X, y, beta=0.8, k=5, weights=feature_weights)
print("After ADASYN :", np.bincount(y_resampled.astype(int)))


# -------------------------
# Train CART + AdaBoost with Regularized Gini
# -------------------------
cart = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,             
    min_samples_split=10,
    min_samples_leaf=5,
    ccp_alpha=0.005,         
    random_state=42
)

ada_cart = AdaBoostClassifier(
    estimator=cart,
    n_estimators=50,         
    learning_rate=0.8,       
    random_state=42
)


print("Starting AdaBoost training...")
start_time = time.time()
ada_cart.fit(X_resampled, y_resampled.astype(int))
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Save the trained model
joblib.dump(ada_cart, 'enhanced_cart_model.joblib')
print("Trained model saved as enhanced_cart_model.joblib")
