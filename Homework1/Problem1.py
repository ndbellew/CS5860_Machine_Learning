import numpy as np
import matplotlib.pyplot as plt

NUM_OF_DATA = 195
RANDOM_SEED = 5860
NORM_VALUE = NUM_OF_DATA / 2
DEGREE = 5
NUM_OF_EPOCHS = 775
BATCH_SIZE = 18
LEARNING_RATE=0.33

def print_polynomial_graph(X_norm: np.ndarray, y: np.ndarray, theta: np.ndarray, true_coefs: np.ndarray):
    # Sort the X_norm values for a smooth curve.
    sorted_idx = np.argsort(X_norm[:, 0])
    x_sorted = X_norm[sorted_idx, 0]

    # Create polynomial objects. np.poly1d expects coefficients in descending order,
    # so we reverse the arrays.
    learned_poly = np.poly1d(theta[::-1])
    true_poly = np.poly1d(true_coefs[::-1])

    # Evaluate the polynomials on the sorted x values.
    y_learned = learned_poly(x_sorted)
    y_true = true_poly(x_sorted)

    # Create the plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(X_norm, y, label="Data", color="grey", alpha=0.5)
    plt.plot(x_sorted, y_true, label="True Polynomial", color="blue", linewidth=2)
    plt.plot(x_sorted, y_learned, label="Learned Polynomial", color="red", linewidth=2)
    plt.xlabel("X (normalized)")
    plt.ylabel("y")
    plt.legend()
    plt.title("Comparison of True and Learned Polynomial")
    plt.show()

def sgd_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100, batch_size: int = 1):
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


    return theta

if __name__ == "__main__":
    randy_savage = np.random.default_rng(RANDOM_SEED)

    X = randy_savage.integers(low=1, high=101, size=(NUM_OF_DATA, 1))

    X_norm = (X - NORM_VALUE) / float(NORM_VALUE)  # Now values are in [-0.98, 1]

    X_poly = np.vander(X_norm[:, 0], N=DEGREE+1, increasing=True)

    
    true_coefs = np.array([0,5,0,-20,0,16])
    y_true = X_poly.dot(true_coefs)
    noise = randy_savage.random(NUM_OF_DATA) * 0.1
    y = y_true + noise
    
    theta = sgd_regression(X_poly, y, lr=LEARNING_RATE, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)
    
    print("Learned theta:")
    print(theta)
    
    true_poly = np.poly1d(true_coefs[::-1])
    print("\nTrue polynomial:")
    print(true_poly)

    sorted_idx = np.argsort(X_norm[:, 0])
    X_sorted = X_norm[sorted_idx, 0]

    learned_poly = np.poly1d(theta[::-1])
    true_poly = np.poly1d(true_coefs[::-1])

    y_learned = learned_poly(X_sorted)
    y_true = true_poly(X_sorted)

    print_polynomial_graph(X_norm, y, theta, true_coefs)