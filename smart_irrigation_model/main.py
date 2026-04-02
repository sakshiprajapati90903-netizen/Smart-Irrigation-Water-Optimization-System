import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data=pd.read_csv("DATASET.csv")
print(data)
print(data.head())
print("\nDataset Info:")
print(data.info)
print("\nMissing Values:")
print(data.isnull().sum().sum())
print(data.head())


data.columns = ['Crop_Type', 'Soil_Type', 'Region', 'Temperature', 'Weather_Condition', 'Water_Requirement']

# Function to convert temperature range into mean (average)
def convert_temp_mean(temp_range):
    low, high = map(int, temp_range.split('-'))
    return (low + high) / 2

# Apply function to Temperature column
data['Temperature'] = data['Temperature'].apply(convert_temp_mean)

# Check result
print(data[['Temperature']].head())

le_crop = LabelEncoder()
le_soil = LabelEncoder()
le_region = LabelEncoder()
le_weather = LabelEncoder()

data['Crop_Type'] = le_crop.fit_transform(data['Crop_Type'])
data['Soil_Type'] = le_soil.fit_transform(data['Soil_Type'])
data['Region'] = le_region.fit_transform(data['Region'])
data['Weather_Condition'] = le_weather.fit_transform(data['Weather_Condition'])

print("\nProcessed Dataset:")
print(data.head())

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

X = data[['Crop_Type', 'Soil_Type', 'Region', 'Temperature', 'Weather_Condition']]
y = data['Water_Requirement']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

results = pd.DataFrame({
    'Actual Water Requirement': y_test.values,
    'Predicted Water Requirement': y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Water Requirement")
plt.ylabel("Predicted Water Requirement")
plt.title("Actual vs Predicted Water Requirement")
plt.show()

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importance)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance")
plt.show()


sample_input = pd.DataFrame({
    'Crop_Type': [le_crop.transform(['BANANA'])[0]],
    'Soil_Type': [le_soil.transform(['DRY'])[0]],
    'Region': [le_region.transform(['DESERT'])[0]],
    'Temperature': [35],   # Example: 30-40 → 35
    'Weather_Condition': [le_weather.transform(['SUNNY'])[0]]
})

predicted_water = model.predict(sample_input)

print("\nSample Prediction:")
print("Predicted Water Requirement for given conditions:", predicted_water[0])


import joblib

joblib.dump(model, "smart_irrigation_model.pkl")
joblib.dump(le_crop, "crop_encoder.pkl")
joblib.dump(le_soil, "soil_encoder.pkl")
joblib.dump(le_region, "region_encoder.pkl")
joblib.dump(le_weather, "weather_encoder.pkl")

print("\nModel and encoders saved successfully!")