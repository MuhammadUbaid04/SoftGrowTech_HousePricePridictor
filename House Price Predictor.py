# ======================================================
#        SIMPLE HOUSE PRICE PREDICTION (ML)
# ======================================================
# Requirements:
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------
# Dataset must contain house features and a "Price" column

df = pd.read_csv("houses.csv")


# ------------------------------------------------------
# 2. DATA PREPROCESSING / CLEANING
# ------------------------------------------------------

# Remove rows with missing values
df = df.dropna()

# Split features and target
X = df.drop("Price", axis=1)
y = df["Price"]


# ------------------------------------------------------
# 3. TRAIN MODEL
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# ------------------------------------------------------
# 4. EVALUATE MODEL
# ------------------------------------------------------

predictions = model.predict(X_test)

accuracy = r2_score(y_test, predictions)
print("Model Accuracy (R²):", round(accuracy, 2))


# ------------------------------------------------------
# 5. PREDICT HOUSE PRICE
# ------------------------------------------------------

print("\nEnter house details:")

size = float(input("Size (sqft): "))
bedrooms = float(input("Bedrooms: "))
bathrooms = float(input("Bathrooms: "))
age = float(input("Age of house: "))
garage = float(input("Garage spaces: "))

house = pd.DataFrame([[size, bedrooms, bathrooms, age, garage]],columns=X.columns)

predicted_price = model.predict(house)

print("\nPredicted House Price: $", round(predicted_price[0], 2))