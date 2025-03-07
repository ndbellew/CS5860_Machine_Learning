from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Homework1.to_Latex import to_latex
from ccpp import get_ccpp_data
from get_ai4i2020_data import get_ai4i_dataset

NUM_OF_EPOCHS = 100
BATCH_SIZE = 64
L2_LAMBDA = 0.1
LEARNING_RATE=0.01

def Ridge(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100, batch_size: int = 1, lmbda: float = 0.1):
    try:
        N, n = X.shape
        theta = np.zeros((X.shape[1], y.shape[1]))
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
                    gradient = (X_batch.T.dot(error) / batch_size) + (lmbda * theta)
                    theta -= lr * gradient
            except Exception as forloop:
                print(f"ERROR {type(forloop).__name__}: has occured during Ridge for loop: {forloop=}\n{start=}\n{N=}\n{X_batch=}\n{y_batch=}\n{theta=}\n{gradient}")
                return None

            epoch_predictions = X.dot(theta)
            epoch_loss = np.mean((epoch_predictions - y) ** 2) / 2 + (lmbda / 2) * np.sum(theta ** 2)  # Ridge Loss
            losses.append(epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
            if np.isnan(epoch_loss):
                raise Exception("Epoch Loss is Nan")
    except Exception as e:
        print(f"ERROR {type(e).__name__}: has occured during Ridge regression: {e=}\n{X=}\n{y}\n{start=}\n{N=}\n{X_batch=}\n{y_batch=}\n{theta=}\n{gradient}")
        return None


    return theta, losses

# def normalization(X,Y):
#     scaler_X = StandardScaler()
#     scaler_Y = StandardScaler()
#     x = scaler_X.fit_transform(X)
#     y = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
#     return  x, y, scaler_Y# X_norm

def graph_it(loss_values, epochs):
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

def main():
    # Goal get X and y, normalize X, send to SGD()
    # Energy Predictions output
    # X,y = get_ccpp_data()
    # X_norm, y_norm, scaler_Y = normalization(X, y)
    # theta, losses = Ridge(X=X_norm, y=y_norm, lr=0.01, epochs=100, batch_size=64, lmbda=0.1)
    # epochs = range(1, NUM_OF_EPOCHS + 1)
    # print(f"Power Plant:\n{to_latex(list(theta))}")
    X, y, = get_ai4i_dataset()
    theta, losses = Ridge(X=X, y=y, lr=LEARNING_RATE, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE,
                          lmbda=L2_LAMBDA)
    epochs = range(1, NUM_OF_EPOCHS + 1)
    print(f"Machine Failure:\n{to_latex(list(theta))}")
    graph_it(losses, epochs)
    # Large-scale Wave Energy Farm





if __name__ == "__main__":
    main()