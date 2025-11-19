# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data (Assuming you downloaded the csv)
# df = pd.read_csv('yield_df.csv') 

# Mock data structure for demonstration
data = {
    'Area': ['India', 'Brazil', 'India', 'Brazil'],
    'Item': ['Rice', 'Coffee', 'Rice', 'Coffee'],
    'average_rain_fall_mm_per_year': [1485, 1761, 1500, 1700],
    'pesticides_tonnes': [120, 150, 130, 140],
    'avg_temp': [24.5, 25.0, 24.8, 25.2],
    'hg/ha_yield': [30000, 20000, 31000, 21000] # Target
}
df = pd.DataFrame(data)

# 2. Preprocessing
# We need to predict 'hg/ha_yield'
X = df.drop(['hg/ha_yield'], axis=1)
y = df['hg/ha_yield']

# Preprocessing pipeline:
# Categorical data (Area, Item) needs OneHotEncoding
# Numerical data needs Scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']),
        ('cat', OneHotEncoder(), ['Area', 'Item'])
    ])

# 3. Define the Model (Random Forest is robust)
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("Training the model...")
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae}") # Lower is better
print(f"R2 Score: {r2}") # Closer to 1 is better

# 7. Visualization (Actual vs Predicted)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yields")
plt.show()