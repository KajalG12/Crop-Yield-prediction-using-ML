# crop_yield_project.py
# Ready for Spyder / local execution

# 0. Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ------------------------------
# 1. Load dataset
# ------------------------------
# change path to where your crop_yield.csv is located
DATA_PATH = r"C:\AI and ML\Projects\Crop Yield Prediction Using ML\crop_yield.csv"
data = pd.read_csv(DATA_PATH)
print("Loaded rows:", data.shape[0], "columns:", data.shape[1])
print(data.head())

# ------------------------------
# 2. Quick data checks
# ------------------------------
print("\n--- Data Info ---")
print(data.info())
print("\n--- Missing values per column ---")
print(data.isnull().sum())

# If there are missing values you can fill or drop as appropriate:
# data = data.dropna()  # or
# data['Fertilizer'].fillna(data['Fertilizer'].median(), inplace=True)

# ------------------------------
# 3. Feature selection & types
# ------------------------------
# Target
TARGET = 'Yield'

# Numeric features you want to use
numeric_features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# Categorical features (will be one-hot encoded)
categorical_features = ['Crop', 'Season', 'State']

# Defensive check: ensure columns exist
for c in numeric_features + categorical_features + [TARGET]:
    if c not in data.columns:
        raise ValueError(f"Column missing in CSV: {c}")

X = data[numeric_features + categorical_features]
y = data[TARGET]

# ------------------------------
# 4. Preprocessing pipelines
# ------------------------------
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # drop any other columns not specified
)

# ------------------------------
# 5. Train / Test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ------------------------------
# 6. Define models inside pipelines
# ------------------------------
models = {
    'LinearRegression': Pipeline(steps=[('preprocessor', preprocessor),
                                        ('regressor', LinearRegression())]),
    'DecisionTree': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', DecisionTreeRegressor(random_state=42))]),
    'RandomForest': Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])
}

# ------------------------------
# 7. Baseline training & CV evaluation
# ------------------------------
results = {}
for name, pipeline in models.items():
    print(f"\nTraining & cross-validating: {name}")
    # 3-fold CV R2 to get a stable sense of performance
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
    print(f"CV R2 scores: {scores}, mean: {scores.mean():.3f}")
    # Fit on full training set to compare on test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {'cv_r2_mean': scores.mean(), 'r2_test': r2, 'rmse_test': rmse, 'mae_test': mae}
    print(f"{name} → Test R2: {r2:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}")

# Summary table
results_df = pd.DataFrame(results).T
print("\nModel comparison:\n", results_df)

# ------------------------------
# 8. Hyperparameter tuning for RandomForest (GridSearch)
# ------------------------------
param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
}

rf_pipeline = models['RandomForest']
print("\nStarting GridSearchCV for RandomForest (this may take a while)...")
grid = GridSearchCV(rf_pipeline, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_rf = grid.best_estimator_

# Evaluate tuned best RF on test set
y_pred_best = best_rf.predict(X_test)
r2_best = r2_score(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_best = mean_absolute_error(y_test, y_pred_best)
print(f"Tuned RandomForest → Test R2: {r2_best:.3f}, RMSE: {rmse_best:.3f}, MAE: {mae_best:.3f}")

# Save the best model (pipeline includes preprocessing)
BEST_MODEL_NAME = 'best_crop_yield_model.pkl'
joblib.dump(best_rf, BEST_MODEL_NAME)
print(f"\nSaved best model to {BEST_MODEL_NAME}")

# ------------------------------
# 9. Feature importance (for tree-based model)
# ------------------------------
# We need the preprocessor to expand categorical feature names
try:
    # sklearn >=1.0: get_feature_names_out
    feature_names_num = numeric_features
    cat_names = list(preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_features))
    feature_names = feature_names_num + cat_names
    importances = best_rf.named_steps['regressor'].feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False).head(20)
    print("\nTop feature importances:\n", fi_df)
    # plot
    plt.figure(figsize=(8,6))
    sns.barplot(x='importance', y='feature', data=fi_df)
    plt.title("Top Feature Importances (Random Forest)")
    plt.show()
except Exception as e:
    print("Could not extract feature names due to scikit-learn version or pipeline structure:", e)

# ------------------------------
# 10. Quick prediction example (use best_rf)
# ------------------------------
# Build a sample input — add zero columns for one-hot if needed
sample = pd.DataFrame({
    'Area': [100],
    'Annual_Rainfall': [750],
    'Fertilizer': [120],
    'Pesticide': [30],
    'Crop': ['Wheat'],
    'Season': ['Rabi'],
    'State': ['Maharashtra']
})
pred_sample = best_rf.predict(sample)
print(f"\nPredicted Yield for sample input: {pred_sample[0]:.2f}")

