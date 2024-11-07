# Demand Forecasting Prediction Model

This project implements a demand forecasting model that combines a Random Forest regressor and a LightGBM model to predict sales for a given dataset. The model first uses the Random Forest to predict sales, then uses the residuals from this prediction to train a LightGBM model, which refines the prediction by accounting for residual errors.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling Approach](#modeling-approach)
6. [Evaluation](#evaluation)
7. [Results](#results)

### Project Structure

The main files in this project are:
- `train.csv`: Training dataset containing historical sales data.
- `test.csv`: Test dataset for which sales predictions will be made.
- `submission.csv`: Output file containing the final sales predictions for the test dataset.

### Features

The model uses the following features:
- `store`: Store identifier
- `item`: Item identifier
- `year`: Extracted from the date
- `month`: Extracted from the date
- `day_of_week`: Extracted from the date

### Installation

To run this project, install the required libraries:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib
```

### Data Preprocessing

Date-related features (`year`, `month`, and `day_of_week`) are extracted from the `date` column in both the training and test datasets. This is done through the `add_date_features` function.

### Modeling Approach

1. **Random Forest Regressor**: Trained on the engineered features to generate initial sales predictions.
2. **LightGBM**: Trained on the residuals (errors) from the Random Forest model, improving prediction accuracy by adjusting for these errors.

### Evaluation

The model is evaluated using Root Mean Squared Error (RMSE) on a validation set. For visualization:
- **Actual vs Predicted Sales**: A plot comparing true sales to predicted sales on the validation set.
- **Prediction Errors**: A histogram showing the distribution of residual errors.
- **Scatter Plot of Actual vs Predicted Sales**: For assessing model fit.
- **Predicted Sales Over Time**: A time series plot showing predicted sales for the test set.

### Results

The final combined model RMSE on the validation set is displayed, providing a measure of model accuracy.

The model will generate `submission.csv` with predicted sales for each item-store combination.
