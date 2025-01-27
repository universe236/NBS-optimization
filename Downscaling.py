import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

train_data = pd.read_excel("training_data.xlsx")
predict_data = pd.read_excel("prediction_data.xlsx")

X = train_data.iloc[:, 1:7]
y = train_data.iloc[:, 7]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5)

y_pred = best_rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Parameters:", grid_search.best_params_)
print("Cross-validation mean R²:", cv_scores.mean())
print("Test set R²:", r2)
print("Test set RMSE:", rmse)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:\n", feature_importance)

predictions = best_rf.predict(predict_data.iloc[:, 1:7])

predict_data['Predicted'] = predictions
predict_data.to_excel("prediction_results.xlsx", index=False)