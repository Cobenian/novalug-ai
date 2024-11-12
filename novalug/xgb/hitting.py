import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from termcolor import cprint


def save_model(reg, df):
    # save the model to a file
    reg.save_model("models/xgb/hitting_model.json")
    # save the dataframe to a file
    df.to_csv("models/xgb/hitting_with_predictions.csv")


def load_model():
    # load the model from a file
    reg = XGBRegressor()
    reg.load_model("models/xgb/hitting_model.json")
    # load the dataframe from a file
    df = pd.read_csv("models/xgb/hitting_with_predictions.csv")
    df = df.drop(columns=["Unnamed: 0"])

    return reg, df


def train():

    cprint("Let's look at predicting the OPS for the hitters", "blue")
    print("")
    print("")
    print("")

    # Load the CSV file into a DataFrame
    df = pd.read_csv("data/stats/hitting.csv")

    # Convert object columns to numeric, forcing errors to NaN
    for col in ["SB%", "PA/BB", "LD%", "FB%", "GB%", "BABIP", "BA/RISP", "AB/HR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Define the features (X) and the target (y)
    X = df.drop(columns=["BATTER", "OPS", "SLG", "OBP"])
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
    cprint("Predictions:", "blue")
    cprint(preds, "yellow")

    # Create a DataFrame to compare actual vs predicted values
    comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})

    # Print the comparison
    cprint("Actual vs predicted values:", "blue")
    cprint(comparison_df, "yellow")

    # calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, preds)
    cprint(f"Mean Squared Error: {mse}", "blue")
    cprint("Done training the model, now let's use it!", "green")

    return reg, df


def make_lineup(reg, df):
    print("")
    print("")
    print("")
    cprint("Now it is time to make our lineup:", "blue")
    print("")
    print("")
    print("")

    X = df.drop(columns=["BATTER", "OPS", "SLG", "OBP"])

    # before we make predications for some of the players b/c we split the test and train data
    # now we need predictions for all the players
    preds = reg.predict(X)
    # print(preds)
    df["POPS"] = preds

    # sort the df by OPS
    df = df.sort_values(by="POPS", ascending=False)

    # print the BATTER from each row in the dataframe
    idx = 1
    for _index, row in df.iterrows():
        if idx <= 9:
            cprint(
                f"Batter {idx}: {row["BATTER"]} (Predict: {row["POPS"]}, Actual OPS: {row["OPS"]})",
                "blue",
            )
        else:
            cprint(
                f"Subs: {row["BATTER"]} (Predict: {row["POPS"]}, Actual OPS: {row["OPS"]})",
                "blue",
            )
        idx += 1


def main():
    reg, df = train()
    save_model(reg, df)
    reg, df = load_model()
    make_lineup(reg, df)
