from pprint import pprint

import pandas as pd


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
    return X, y
