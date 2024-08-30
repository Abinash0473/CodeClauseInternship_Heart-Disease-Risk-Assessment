import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
data = pd.read_csv('Heart_Disease_Prediction.csv')

# Preprocess data
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()

# Create a Random Forest pipeline with scaling and hyperparameter tuning
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', RandomForestClassifier(random_state=42))
])

# Define hyperparameters to tune
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Improved model accuracy: {accuracy * 100:.2f}%")

# Save the improved model to a file
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
