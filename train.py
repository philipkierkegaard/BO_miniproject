import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint
import time

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Split Data
X = df.drop(columns=['Quality'])  # Features
y = df['Quality']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ BASELINE MODEL ------------------
print("\n==== BASELINE MODEL ====")
baseline_model = RandomForestClassifier(random_state=42)
baseline_start_time = time.time()
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - baseline_start_time

baseline_preds = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_preds)

print("Baseline Accuracy:", baseline_acc)
print("Baseline Classification Report:\n", classification_report(y_test, baseline_preds))

# ------------------ RANDOM SEARCH ------------------
print("\n==== RANDOM SEARCH ====")

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_start_time = time.time()
random_search.fit(X_train, y_train)
random_time = time.time() - random_start_time

random_best_model = random_search.best_estimator_
random_preds = random_best_model.predict(X_test)
random_acc = accuracy_score(y_test, random_preds)

print("Random Search Best Params:", random_search.best_params_)
print("Random Search Accuracy:", random_acc)
print("Random Search Classification Report:\n", classification_report(y_test, random_preds))

# ------------------ BAYESIAN OPTIMIZATION (Optuna) ------------------
print("\n==== BAYESIAN OPTIMIZATION ====")

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
    
    return accuracy

bo_start_time = time.time()
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
bo_time = time.time() - bo_start_time

best_params = study.best_params
print("Bayesian Optimization Best Params:", best_params)

bo_best_model = RandomForestClassifier(**best_params, random_state=42)
bo_best_model.fit(X_train, y_train)
bo_preds = bo_best_model.predict(X_test)
bo_acc = accuracy_score(y_test, bo_preds)

print("Bayesian Optimization Accuracy:", bo_acc)
print("Bayesian Optimization Classification Report:\n", classification_report(y_test, bo_preds))

# ------------------ COMPARISON PLOT ------------------

models = ["Baseline", "Random Search", "Bayesian Optimization"]
accuracies = [baseline_acc, random_acc, bo_acc]
times = [baseline_time, random_time, bo_time]

plt.figure(figsize=(10, 5))

# Accuracy Comparison
plt.subplot(1, 2, 1)
sns.barplot(x=models, y=accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")

# Time Comparison
plt.subplot(1, 2, 2)
sns.barplot(x=models, y=times)
plt.ylabel("Time (seconds)")
plt.title("Time Taken for Optimization")

plt.tight_layout()
plt.show()
