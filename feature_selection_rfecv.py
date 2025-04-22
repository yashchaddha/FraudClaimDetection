import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with your actual data file)
# Assuming you have a CSV file with features and target variable
data = pd.read_csv('your_data.csv')

# Separate features and target
# Replace 'target_column' with your actual target column name
X = data.drop('target_column', axis=1)
y = data['target_column']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Create the RFECV object
# We'll use LogisticRegression as the estimator
estimator = LogisticRegression(max_iter=1000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=cv,
    scoring='accuracy',
    min_features_to_select=1,
    n_jobs=-1
)

# Fit RFECV
rfecv.fit(X_scaled, y)

# Get selected features
selected_features = X.columns[rfecv.support_]
ranking = pd.Series(rfecv.ranking_, index=X.columns)

# Print results
print("\nOptimal number of features:", rfecv.n_features_)
print("\nSelected features:")
print(selected_features.tolist())
print("\nFeature ranking (1 indicates selected features):")
print(ranking)

# Plot number of features vs cross-validation scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel('Number of features selected')
plt.ylabel('Cross-validation score')
plt.title('Feature Selection using RFECV')
plt.grid(True)
plt.savefig('rfecv_scores.png')
plt.close()

# Create a feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Selected': rfecv.support_,
    'Ranking': rfecv.ranking_
})
feature_importance = feature_importance.sort_values('Ranking')
print("\nFeature importance summary:")
print(feature_importance)

# Save selected features to a file
with open('selected_features.txt', 'w') as f:
    f.write("Optimal number of features: {}\n\n".format(rfecv.n_features_))
    f.write("Selected features:\n")
    for feature in selected_features:
        f.write(f"{feature}\n")