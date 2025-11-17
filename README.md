🌾 Crop Yield Prediction Using Machine Learning
A Data-Driven ML System for Smart Agriculture
---

📌 Overview :

This project predicts crop yield using real agricultural parameters such as rainfall, fertilizer usage, area, season, crop type, and pesticide consumption.
It is built using Python, Pandas, Scikit-Learn, and Machine Learning pipelines.

---

The system helps in:

1) Estimating expected yield

2) Identifying factors affecting productivity

3) Supporting farmers with data-driven decisions

4) Generating predictions for new crop inputs
---
🚀 Features

✔ Data loading, cleaning and preprocessing
✔ Exploratory Data Analysis (EDA)
✔ ML model training (Linear Regression, Decision Tree, Random Forest)
✔ Cross-validation & evaluation metrics
✔ Full preprocessing + model pipeline
✔ Hyperparameter tuning with GridSearchCV
✔ Feature importance visualization
✔ Saving best model as .pkl
✔ Single-input yield prediction

---

| Column Name     | Description                    |
| --------------- | ------------------------------ |
| Crop            | Crop type (e.g., Wheat, Rice)  |
| Crop_Year       | Year of cultivation            |
| Season          | Rabi / Kharif / Summer         |
| State           | Indian state name              |
| Area            | Area of land (in hectares)     |
| Production      | Total production (tonnes)      |
| Annual_Rainfall | Rainfall (mm)                  |
| Fertilizer      | Fertilizer consumption         |
| Pesticide       | Pesticide usage                |
| Yield           | Yield output (Target variable) |

---

🧪 Technologies Used

- Python 3

- Pandas

- NumPy

- Matplotlib + Seaborn

- Scikit-Learn

- Joblib

- Random Forest Regressor
  
---

📉 Modeling Approach
1. Data Preprocessing

- Standard scaling for numerical features

- One-Hot Encoding for categorical features

- ColumnTransformer + Pipeline for clean workflow

2. Models Used

- Linear Regression

- Decision Tree Regressor

- Random Forest Regressor (Best Model)

3. Model Evaluation Metrics

- R² Score

- RMSE (Root Mean Squared Error)

- MAE (Mean Absolute Error)

4. Hyperparameter Tuning

- GridSearchCV used for:

- n_estimators

- max_depth

- min_samples_split

---

📊 Feature Importance Plot
A Random Forest–based feature importance graph is generated to show which features influence yield most.

![Random Forest](images/feature_importance.png)

---

🌐 Future Enhancements

- Add deep learning models for disease detection

- Use satellite imagery (NDVI)

- Deploy the model using Streamlit

- Convert into a full agriculture-advisory system

- Add soil nutrient (NPK) integration

---

## 📟 Console Output (Training Logs)

```
Loaded rows: 19689 columns: 10
           Crop       Crop_Year       Season       ...  Fertilizer      Pesticide     Yield
0      Arecanut       1997            Whole Year   ...  7024878.38      22882.34      0.796087
1     Arhar/Tur       1997            Kharif       ...   631643.29      2057.47       0.710435
2   Castor seed       1997            Kharif       ...    75755.32      246.76        0.238333
3      Coconut        1997            Whole Year   ...  1870661.52      6093.36       5238.051739
4  Cotton(lint)       1997            Kharif       ...   165500.63      539.09        0.420909
[5 rows x 10 columns]

--- Data Info ---
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19689 entries
Columns: 10

Data columns (total 10 columns):
 #   Column           Non-Null Count    Dtype  
 0   Crop             19689 non-null    object 
 1   Crop_Year        19689 non-null    int64  
 2   Season           19689 non-null    object 
 3   State            19689 non-null    object 
 4   Area             19689 non-null    float64
 5   Production       19689 non-null    int64  
 6   Annual_Rainfall  19689 non-null    float64
 7   Fertilizer       19689 non-null    float64
 8   Pesticide        19689 non-null    float64
 9   Yield            19689 non-null    float64
dtypes: float64(5), int64(2), object(3)
memory usage: 1.5+ MB

--- Missing values ---
Crop               0
Crop_Year          0
Season             0
State              0
Area               0
Production         0
Annual_Rainfall    0
Fertilizer         0
Pesticide          0  
Yield              0

Train size: 15751, Test size: 3938

Training & cross-validating: LinearRegression
CV R2 scores: [0.846 0.844 0.858], mean: 0.850
LinearRegression → Test R2: 0.810, RMSE: 389.719, MAE: 63.216

Training & cross-validating: DecisionTree
CV R2 scores: [0.844 0.968 0.956], mean: 0.923
DecisionTree → Test R2: 0.970, RMSE: 156.169, MAE: 10.103

Training & cross-validating: RandomForest
CV R2 scores: [0.882 0.949 0.974], mean: 0.936
RandomForest → Test R2: 0.980, RMSE: 125.046, MAE: 8.959

Model comparison:
                   cv_r2_mean   r2_test   rmse_test   mae_test
LinearRegression    0.850       0.810     389.719     63.216
DecisionTree        0.923       0.970     156.169     10.103
RandomForest        0.936       0.980     125.046      8.959

Starting GridSearchCV...
Best params:
{
  'regressor__n_estimators': 100,
  'regressor__max_depth': 10,
  'regressor__min_samples_split': 5
}

Tuned RandomForest → Test R2: 0.979
RMSE: 128.76
MAE: 9.80

Top feature importances:
              feature        importance
12          Crop_Coconut     0.8491
1        Annual_Rainfall     0.0331
74       State_Karnataka     0.0212
3              Pesticide     0.0206
0                   Area     0.0202
2             Fertilizer     0.0181
91     State_West Bengal     0.012574
64           State_Assam     0.010989
66    State_Chhattisgarh     0.004906
87       State_Telangana     0.004151
83      State_Puducherry     0.002024
86      State_Tamil Nadu     0.001256
68             State_Goa     0.000952
61     Season_Whole Year     0.000270
49        Crop_Sugarcane     0.000094
58         Season_Kharif     0.000061
75          State_Kerala     0.000057
80         State_Mizoram     0.000027
77     State_Maharashtra     0.000018
6            Crop_Banana     0.000009

...

Predicted Yield for sample input: 297.92

Model saved as best_crop_yield_model.pkl

```

🎉 Final Result

- Achieved 98% Test Accuracy (R² = 0.980)

- Identified Coconut + Rainfall + State as top influencers

- Successfully predicts yield for new inputs

- Saves the best model for deployment

---  
✔ Saving best model as
.pkl
✔ Single-input yield prediction
