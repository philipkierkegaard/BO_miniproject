import numpy as np
from skopt.space import Real, Integer, Categorical
import skopt.space
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold, cross_val_score


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

#search_space = [skopt.space.Real(-3,3, name='x1'), skopt.space.Real(-3,3, name='x2')]

#hyperparameters search
search_space = [
    Integer(10, 200, name="n_estimators"),          
    Integer(1, 20, name="max_depth"),               
    Categorical(["gini", "entropy"], name="criterion"),  
    Categorical(["sqrt", "log2"], name="max_features") 
]
#rs = RandomizedSearchCV(model, param_distributions=domain, cv=3, verbose =2, n_iter=10)
#rs.fit(Xtrain, ytrain)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
@use_named_args(search_space)
def objective(n_estimators, criterion, max_depth, max_features):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    # Use cross-validation
    score = np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy"))
    return -score  # Minimizing

# Run Bayesian Optimization
bayopt_start_time = time.time()
result = gp_minimize(objective, search_space, n_calls=20, random_state=42)
bayopt_time = time.time() - bayopt_start_time

print("Best Hyperparameters:")
print(f"n_estimators: {result.x[0]}")
print(f"max_depth: {result.x[1]}")
print(f"criterion: {result.x[2]}")
print(f"Best cross-validation accuracy: {-result.fun:.4f}")

# Train Best model
best_model = RandomForestClassifier(
    n_estimators=result.x[0],
    max_depth=result.x[1],
    criterion=result.x[2],
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# Eval
test_accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

# 1️⃣ Convergence Plot: Shows how accuracy improves over iterations
plot_convergence(result)
plt.show()

# 2️⃣ Partial Dependence Plots: Shows how each hyperparameter affects performance
plot_objective(result)
plt.show()

# 3️⃣ Search Space Exploration: Displays parameter combinations that were tested
plot_evaluations(result)
plt.show()

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
    cv=3,
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



# ------------------ COMPARISON PLOT ------------------

models = ["Baseline", "Random Search", "Bayesian Optimization"]
accuracies = [baseline_acc, random_acc, test_accuracy]
times = [baseline_time, random_time, bayopt_time]

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
