from pprint import pprint

import pandas as pd
from Homework1.Problem2.Problem2 import SGD, graph_it, normalization

NUM_OF_EPOCHS = 84
BATCH_SIZE = 16
LEARNING_RATE=0.001

def clean_energy_data(dataframe):
    # the timestampe: "date" is in datetime form and i need to work with numbers only so I need to clean the data
    dataframe["date"] = pd.to_datetime(dataframe["date"])
    the_epoch = pd.Timestamp("1970-01-01")
    dataframe["date"] = (dataframe["date"] - the_epoch).dt.total_seconds().astype(int)
    return dataframe

def read_energy_dataset(filename: str = "energydata_complete.csv"):
    dataframe = pd.read_csv(filename)
    feature_names = dataframe.columns[dataframe.columns != "Appliances"]
    dataframe = clean_energy_data(dataframe)

    X = dataframe.drop(columns=["Appliances"]).to_numpy()
    y = dataframe["Appliances"].to_numpy()
    return X, y, feature_names

def main():
    # Goal get X and y, normalize X, send to SGD()
    # Energy Predictions output
    X, y, names = read_energy_dataset()
    X_norm, y_norm, _ = normalization(X, y)
    theta, losses = SGD(X=X_norm, y=y_norm, lr=LEARNING_RATE, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)
    epochs = range(1, NUM_OF_EPOCHS + 1)
    pprint(theta)
    graph_it(losses, epochs)

    if __name__ == "__main__":
        main()