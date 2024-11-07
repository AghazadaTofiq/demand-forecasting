import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])

# Feature Engineering
def add_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    return df

train = add_date_features(train)
test = add_date_features(test)

features = ['store', 'item', 'year', 'month', 'day_of_week']
X = train[features]
y = train['sales']

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Residuals for LightGBM training
y_pred_rf = rf_model.predict(X_val)
residuals = y_val - y_pred_rf

# LightGBM Model
lgb_train = lgb.Dataset(X_val, label=residuals)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

# Combined Prediction on Validation Set
rf_val_pred = rf_model.predict(X_val)
lgb_val_pred = lgb_model.predict(X_val)
combined_val_pred = rf_val_pred + lgb_val_pred

# Model Evaluation
rmse = np.sqrt(mean_squared_error(y_val, combined_val_pred))
print(f"Combined Model RMSE: {rmse}")

# Predictios
X_test = test[features]

rf_test_pred = rf_model.predict(X_test)
lgb_test_pred = lgb_model.predict(X_test)

test['sales'] = rf_test_pred + lgb_test_pred

# Save the Predictions
test[['id', 'sales']].to_csv('submission.csv', index=False)

# Plotting Graphs
plt.figure(figsize=(12, 6))
plt.plot(y_val.values, label="Actual Sales", color="blue", marker='o', linestyle='-')
plt.plot(combined_val_pred, label="Predicted Sales (Combined Model)", color="red", marker='x', linestyle='--')
plt.xlabel("Data Point")
plt.ylabel("Sales")
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.show()

# Calculate residuals (errors)
errors = y_val - combined_val_pred

plt.figure(figsize=(10, 5))
plt.hist(errors, bins=20, color="purple", edgecolor="black")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_val, combined_val_pred, alpha=0.5, color="green")
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color="red", linestyle="--")  # 45-degree line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales (Scatter Plot)")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test['date'], test['sales'], label="Predicted Sales (Test Set)", color="orange", marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Predicted Sales on Test Data")
plt.legend()
plt.xticks(rotation=45)
plt.show()
