from pprint import pprint
from ucimlrepo import fetch_ucirepo
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd


NUM_OF_EPOCHS = 30
BATCH_SIZE = 120
LEARNING_RATE=0.06

def SGD(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100, batch_size: int = 1):
    try:
        N, n = X.shape
        theta = np.zeros(n)
        losses = list()

        for epoch in range(epochs):
            indices = np.random.permutation(N)
            try:
                for start in range(0, N, batch_size):
                    batch_idx = indices[start:start+batch_size]

                    X_batch = X[batch_idx]
                    y_batch = y[batch_idx]
                    predictions = X_batch.dot(theta)
                    error = predictions - y_batch
                    gradient = X_batch.T.dot(error) / batch_size
                    theta -= lr * gradient
            except Exception as forloop:
                print(f"ERROR {type(forloop).__name__}: has occured during SGD for loop: {forloop=}\n{start=}\n{N=}\n{X_batch=}\n{y_batch=}\n{theta=}\n{gradient}")
                return None

            epoch_predictions = X.dot(theta)
            epoch_loss = np.mean((epoch_predictions - y) ** 2) / 2  # using 1/2 factor for convenience
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
            if np.isnan(epoch_loss):
                raise Exception("Epoch Loss is Nan")
    except Exception as e:
        print(f"ERROR {type(e).__name__}: has occured during SGD regression: {e=}\n{X=}\n{y}\n{start=}\n{N=}\n{X_batch=}\n{y_batch=}\n{theta=}\n{gradient}")
        return None


    return theta, losses

def normalization(X,Y):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    x = scaler_X.fit_transform(X)
    y = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
    return  x, y, scaler_Y# X_norm

def graph_it(loss_values, epochs):
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

def read_wave_energy_dataset(filenames: str, file_num: int = 1):
    dataframe = None
    for filename in filenames.split():
        if dataframe is not None:
            dataframe = pd.read_csv(filename, delimiter=",")
        else:
            partial_frame = pd.read_csv(filename, delimiter=',')
            dataframe = pd.concat([dataframe, partial_frame], ignore_index=True)
    print(dataframe)
    features_names = dataframe.columns[dataframe.columns != "Total_Power"]

    X = dataframe.drop(columns=["Total_Power"]).to_numpy()
    y = dataframe["Total_Power"].to_numpy()

    return X, y, features_names



def main():
    # Goal get X and y, normalize X, send to SGD()
    # Energy Predictions output
    # fetch dataset
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)

    # data (as pandas dataframes)
    X = ai4i_2020_predictive_maintenance_dataset.data.features
    y = ai4i_2020_predictive_maintenance_dataset.data.targets
    X_norm, y_norm, scaler_Y = normalization(X, y)
    theta, losses = SGD(X=X_norm, y=y_norm, lr=LEARNING_RATE, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)
    epochs = range(1, NUM_OF_EPOCHS + 1)
    pprint(theta)
    graph_it(losses, epochs)
    # Large-scale Wave Energy Farm


if __name__ == "__main__":
    main()