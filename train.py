import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint
from skopt import gp_minimize
from skopt.space import Integer

# Load dataset
df = pd.read_csv('data/dataset.csv')

# Ensure Quality is treated as a continuous variable
df['Quality'] = df['Quality'].astype(float)

# Split Data
X = df.drop(columns=['Quality'])  
y = df['Quality']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Helper Functions ------------------

def random_search_run():
    """ Run Random Search once and return best regression metrics. """
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 10)
    }

    random_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='neg_mean_squared_error',  # Regression: Minimize MSE
        random_state=None,  # No fixed seed to allow variation
        n_jobs=-1
    )

    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_model = random_search.best_estimator_
    best_preds = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, best_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, best_preds)

    return mse, rmse, r2, elapsed_time


def bo_run():
    """ Run Bayesian Optimization once and return best regression metrics. """
    def objective(params):
        n_estimators, max_depth = params
        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=42
        )
        
        mse = -np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
        
        return mse  # Minimize MSE

    search_space = [
        Integer(50, 300, name="n_estimators"),
        Integer(3, 20, name="max_depth")
    ]

    start_time = time.time()
    gp_result = gp_minimize(
        objective,
        search_space,
        acq_func="EI",
        n_calls=20,
        n_random_starts=5,
        random_state=None  # No fixed seed to allow variation
    )
    elapsed_time = time.time() - start_time

    # Train final model with best parameters
    best_n_estimators = gp_result.x[0]
    best_max_depth = gp_result.x[1]
    best_model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
    best_model.fit(X_train, y_train)
    best_preds = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, best_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, best_preds)

    return mse, rmse, r2, elapsed_time


# ------------------ Run Multiple Trials ------------------

num_trials = 10  # Run both methods 10 times each
random_mse, random_rmse, random_r2, random_times = [], [], [], []
bo_mse, bo_rmse, bo_r2, bo_times = [], [], [], []

for i in range(num_trials):
    print(f"Running Trial {i+1}/{num_trials}...")

    # Run Random Search
    mse, rmse, r2, elapsed = random_search_run()
    random_mse.append(mse)
    random_rmse.append(rmse)
    random_r2.append(r2)
    random_times.append(elapsed)

    # Run Bayesian Optimization
    mse, rmse, r2, elapsed = bo_run()
    bo_mse.append(mse)
    bo_rmse.append(rmse)
    bo_r2.append(r2)
    bo_times.append(elapsed)


# ------------------ Compare Results ------------------

# Convert to DataFrame for better visualization
results_df = pd.DataFrame({
    "Random Search MSE": random_mse,
    "Bayesian Optimization MSE": bo_mse,
    "Random Search RMSE": random_rmse,
    "Bayesian Optimization RMSE": bo_rmse,
    "Random Search R²": random_r2,
    "Bayesian Optimization R²": bo_r2,
    "Random Search Time (s)": random_times,
    "Bayesian Optimization Time (s)": bo_times
})

print("\n====== Comparison Results ======")
print(results_df.describe())  # Show mean, std, min, max of results

# Boxplot to visualize MSE distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=results_df[["Random Search MSE", "Bayesian Optimization MSE"]])
plt.title("MSE Distribution: Random Search vs Bayesian Optimization")
plt.ylabel("Mean Squared Error (MSE)")
plt.show()

# Boxplot to visualize RMSE distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=results_df[["Random Search RMSE", "Bayesian Optimization RMSE"]])
plt.title("RMSE Distribution: Random Search vs Bayesian Optimization")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.show()

# Boxplot to visualize R² Score distribution
plt.figure(figsize=(8, 5))
sns.boxplot(data=results_df[["Random Search R²", "Bayesian Optimization R²"]])
plt.title("R² Score: Random Search vs Bayesian Optimization")
plt.ylabel("R² Score (Higher is better)")
plt.show()

# Boxplot to visualize execution time
plt.figure(figsize=(8, 5))
sns.boxplot(data=results_df[["Random Search Time (s)", "Bayesian Optimization Time (s)"]])
plt.title("Execution Time: Random Search vs Bayesian Optimization")
plt.ylabel("Time (seconds)")
plt.show()
