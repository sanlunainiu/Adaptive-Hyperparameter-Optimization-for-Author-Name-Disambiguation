import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from generate_feature_for_rf import generate_training_features
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

author_features = generate_training_features()

features_df = pd.DataFrame.from_dict(author_features, orient='index')

# Read the CSV file containing optimal hyperparameters
# for pf1
# hyperparameters_df = pd.read_csv(r"out\YOUR_pf1_hyperparameter_optimization_result.csv")
# for k_metric
# hyperparameters_df = pd.read_csv(r"out\YOUR_k_hyperparameter_optimization_result.csv")
# for b3
# hyperparameters_df = pd.read_csv(r"out\YOUR_b3_hyperparameter_optimization_result.csv")

# Ensure features_df and hyperparameters_df are aligned in the same order if necessary
combined_df = features_df.merge(hyperparameters_df, left_index=True, right_on='Name')

X = combined_df.drop(['Learning Rate', 'Name', 'prec', 'rec', 'f1'], axis=1).values
y = combined_df['Learning Rate'].values
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regressor model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# train 
regressor.fit(X_train_scaled, y_train)

# predict
y_pred = regressor.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"mse: {mse}")

rmse = mse ** 0.5
print(f"rmse: {rmse}")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# Rank correlation coefficients
spearman_corr, spearman_p = spearmanr(y_test, y_pred)
print(f"Spearman's rank correlation coefficient: {spearman_corr}, p-value: {spearman_p}")
kendall_corr, kendall_p = kendalltau(y_test, y_pred)
print(f"Kendall's tau correlation coefficient: {kendall_corr}, p-value: {kendall_p}")

# feature importances
feature_importances = regressor.feature_importances_
feature_names = combined_df.drop(['Learning Rate', 'Name', 'prec', 'rec', 'f1'], axis=1).columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

importance_df['Percentage'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)

# Visualize feature importance percentages
plt.figure(figsize=(12, 10))
bars = plt.barh(importance_df['Feature'], importance_df['Percentage'])
plt.xlabel('Feature Importance Percentage (%)')
plt.ylabel('Features')

# plt.title('Feature Importance Percentage from Random Forest Regressor')
plt.gca().invert_yaxis()
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%', ha='left', va='center')

plt.tight_layout()
# plt.savefig(r'feature_importance.tif', format='tif', dpi=1200)
plt.show()
