import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load the CSV file into a DataFrame
df = pd.read_csv("data/stats/hitting.csv")
df = df.drop(columns=["BATTER"])

# Convert object columns to numeric, forcing errors to NaN
for col in ["SB%", "PA/BB", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "AB/HR"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Define the features (X) and the target (y)
X = df.drop(columns=["OPS"])
# X = df.drop(columns=["OPS", "SLG", "OBP"])
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


st.title("Hitting Data Predictions")

comparison_df["Actual"]

st.write(comparison_df)

st.table(comparison_df)

# comparison_array = comparison_df.values.tolist()
# comparison_array
# st.line_chart(comparison_array)

# chart_data = pd.DataFrame(
#     comparison_df['Predicted'].to_list(),
#     columns=['Predicted'])
# st.line_chart(chart_data)

# chart_data = pd.DataFrame(
#     comparison_df,
#     columns=['row', 'Actual', 'Predicted'])
# st.line_chart(chart_data)

# st.bar_chart(comparison_df['Predicted'].to_list())

st.line_chart(comparison_df)

edited_comparison_df = st.data_editor(comparison_df)

st.line_chart(edited_comparison_df)

# comparison_df

# preds

# y_test
