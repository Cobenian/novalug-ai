import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV file into a DataFrame
df = pd.read_csv("data/hitting.csv")

# Convert object columns to numeric, forcing errors to NaN
for col in ["SB%", "PA/BB", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "AB/HR"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Define the features (X) and the target (y)
# X = df.drop(columns=["OPS"])
X = df.drop(columns=["OPS", "SLG", "OBP"])
y = df["OPS"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the XGBRegressor model
reg = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1, objective="reg:squarederror"
)

# Train the model on the training data
reg.fit(X_train, y_train)

# Make predictions on the test data
preds = reg.predict(X_test)

# Print the predictions
print("Predictions:")
print(preds)


# Create a DataFrame to compare actual vs predicted values
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})

# Print the comparison
print(comparison_df)

# Optionally, calculate and print the Mean Squared Error
mse = mean_squared_error(y_test, preds)
print(f"Mean Squared Error: {mse}")
