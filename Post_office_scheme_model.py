import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv

import os

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')


# Connect to MongoDB and retrieve data
client = MongoClient(MONGO_URI)
db = client['Model_train']
collection = db['city_data']

# Load data from MongoDB
data = pd.DataFrame(list(collection.find()))

schemes_success = pd.json_normalize(data['schemes_success'])
data = data.drop(columns=['schemes_success'])
data = pd.concat([data, schemes_success], axis=1)

# Define features and target variables
features = ['avg_age', 'gender_ratio', 'avg_income', 'employment_rate', 'farming_cycle']
targets = [
    'SB', 'RD', 'TD', 'MIS',
    'SCSS', 'PPF', 'SSA', 'NSC',
    'KVP', 'Mahila_Samman', 'PM_CARES'
]
print("Data base connected")
print(data.columns)


X = data[features]
y = data[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'post_office_success_model.pkl')

# Predict on the test set
test_predictions = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, test_predictions)
r2 = r2_score(y_test, test_predictions)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'post_office_success_best_model.pkl')

# Load the trained model
model = joblib.load('post_office_success_best_model.pkl')

# Example: Predict for a new city
new_city = {
    "avg_age": 100,
    "gender_ratio": 1.05,
    "avg_income": 45000,
    "employment_rate": 0.85,
    "farming_cycle": 0.6
}


new_city_features = pd.DataFrame([new_city], columns=features)

# Predict success probabilities

probabilities = model.predict(new_city_features)[0]

# Prepare the response
response = {
    "SB": probabilities[0],
    "RD": probabilities[1],
    "TD": probabilities[2],
    "MIS": probabilities[3],
    "SCSS": probabilities[4],
    "PPF": probabilities[5],
    "SSA": probabilities[6],
    "NSC": probabilities[7],
    "KVP": probabilities[8],
    "Mahila_Samman": probabilities[9],
    "PM_CARES": probabilities[10]
}

# Convert np.float64 to native Python float
response = {key: float(value) for key, value in response.items()}
print(response)


