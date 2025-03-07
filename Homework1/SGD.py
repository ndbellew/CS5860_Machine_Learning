
import numpy as np

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
