import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# --- SOP 1: WEIGHTED ADASYN IMPLEMENTATION ---
def weighted_adasyn(X, y, beta=0.8, k=5, weights=None, random_state=None):
    np.random.seed(random_state)
    class_counts = Counter(y)
    major_class = max(class_counts, key=class_counts.get)
    minor_class = min(class_counts, key=class_counts.get)
    n_major, n_minor = class_counts[major_class], class_counts[minor_class]

    if n_minor >= n_major * beta: return X.copy(), y.copy()
    G = int(n_major * beta) - n_minor
    X_min = X[y == minor_class]
    
    if weights is not None:
        sqrt_weights = np.sqrt(weights)
        X_scaled, X_min_scaled = X * sqrt_weights, X_min * sqrt_weights
    else:
        X_scaled, X_min_scaled = X, X_min
        
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    _, indices = nn.kneighbors(X_min_scaled)
    ri_list = [np.sum(y[indices[i, 1:]] == major_class) / k for i in range(n_minor)]
    ri_array = np.array(ri_list)
    sum_ri = np.sum(ri_array)
    ri_hat = ri_array / sum_ri if sum_ri > 0 else np.full(len(ri_array), 1/len(ri_array))
    gi_array = np.round(G * ri_hat).astype(int)

    X_synthetic = []
    nn_minority = NearestNeighbors(n_neighbors=k + 1).fit(X_min_scaled)
    for i in range(n_minor):
        n_to_gen = gi_array[i]
        if n_to_gen > 0:
            _, min_idx = nn_minority.kneighbors([X_min_scaled[i]])
            choices = np.random.choice(min_idx[0, 1:], n_to_gen, replace=True)
            for idx in choices:
                X_synthetic.append(X_min[i] + np.random.rand() * (X_min[idx] - X_min[i]))
            
    return np.vstack([X, np.array(X_synthetic)]), np.concatenate([y, np.full(len(X_synthetic), minor_class)])

model_filename = 'enhanced_model.joblib'
if os.path.exists(model_filename):
    os.remove(model_filename)
    print(f"Removed old version of {model_filename}")

# --- DATA PREPROCESSING ---
df = pd.read_csv("onlinefraud.csv")

features_for_ui = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
X = df[features_for_ui].values
y = df["isFraud"].values

print("Before ADASYN:", np.bincount(y))
feature_weights = np.ones(X.shape[1]) 
X_resampled, y_resampled = weighted_adasyn(X, y, beta=0.8, k=5, weights=feature_weights, random_state=42)
print("After ADASYN :", np.bincount(y_resampled.astype(int)))

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, 
    y_resampled, 
    test_size=0.3, 
    stratify=y_resampled, 
    random_state=42)

# --- SOP 2: REGULARIZED GINI INDEX ---
cart = DecisionTreeClassifier(
    criterion="gini", 
    max_depth=5, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    ccp_alpha=0.001, # used to approximate regularized gini
    random_state=42)

# --- SOP 3: ADABOOST ---
ada_cart = AdaBoostClassifier(
    estimator=cart, 
    n_estimators=50, 
    learning_rate=0.5, 
    random_state=42)

print("\nTraining model on 5 features...")
ada_cart.fit(X_train, y_train)

# --- EVALUATION ---
y_pred = ada_cart.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- SAVE AND VERIFY ---
joblib.dump(ada_cart, model_filename)

verified_model = joblib.load(model_filename)
print("\n" + "="*30)
print(f"VERIFICATION SUCCESSFUL")
print(f"Model saved as: {model_filename}")
print(f"Features expected: {verified_model.n_features_in_}")
print(f"Feature List: {features_for_ui}")
print("="*30)
