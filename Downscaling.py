import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

train_data = pd.read_excel("training_data.xlsx")
predict_data = pd.read_excel("prediction_data.xlsx")

target_col = train_data.columns[-1]
feature_cols = train_data.columns[1:-1]

X = train_data[feature_cols]
y = train_data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Cross-validation mean R2:", grid_search.best_score_)

y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Test set R2:", r2)
print("Test set RMSE:", rmse)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:\n", feature_importance)

if all(col in predict_data.columns for col in feature_cols):
    X_predict = predict_data[feature_cols]
    predictions = best_rf.predict(X_predict)

    predict_data['Predicted'] = predictions
    predict_data.to_excel("prediction_results.xlsx", index=False)
    print("\nPrediction results saved to prediction_results.xlsx")
else:
    print("\nError: Prediction data missing required feature columns.")
