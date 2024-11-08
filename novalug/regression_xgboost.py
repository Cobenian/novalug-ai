import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

print("load data")
data = load_diabetes()

# Convert the data to a DataFrame
df = pd.DataFrame(data=data["data"], columns=data["feature_names"])
df["target"] = data["target"]

print("Data as DataFrame:")
print(df.head())  # Print the first 5 rows of the DataFrame


print("split data")
X_train, X_test, y_train, y_test = train_test_split(
    data["data"], data["target"], test_size=0.2
)

# create model instance
print("create regressor model instance")
reg = XGBRegressor(
    n_estimators=2, max_depth=2, learning_rate=1, objective="reg:squarederror"
)
# fit model
print("fit the model")
reg.fit(X_train, y_train)
# make predictions
print("make predictions")
preds = reg.predict(X_test)
print(preds)
