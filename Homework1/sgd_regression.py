import numpy as np
import matplotlib.pyplot as plt

NUM_OF_DATA = 195
RANDOM_SEED = 5860
NORM_VALUE = NUM_OF_DATA / 2
DEGREE = 5
NUM_OF_EPOCHS = 775
BATCH_SIZE = 18
LEARNING_RATE=0.33

def sgd_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 100, batch_size: int = 1):
    """
    Perform linear regression using stochastic gradient descent (SGD).
    
    Parameters:
        X (np.ndarray): Input feature matrix of shape (N, n).
        y (np.ndarray): Target values vector of shape (N,).
        lr (float): Learning rate.
        epochs (int): Number of epochs to run SGD.
        batch_size (int): Size of mini-batches.
    
    Returns:
        np.ndarray: Learned parameter vector theta.
    """
    N, n = X.shape
    theta = np.zeros(n)
    losses = list()
    
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            predictions = X_batch.dot(theta)
            error = predictions - y_batch
            gradient = X_batch.T.dot(error) / batch_size
            theta -= lr * gradient

        epoch_predictions = X.dot(theta)
        epoch_loss = np.mean((epoch_predictions - y) ** 2) / 2  # using 1/2 factor for convenience
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

    return theta

if __name__ == "__main__":

    # Create a random number generator with a fixed seed
    randy_savage = np.random.default_rng(RANDOM_SEED)
    
    # Generate a 100x1 array of random integers between 1 and 100 (inclusive) rows
    X = randy_savage.integers(low=1, high=101, size=(NUM_OF_DATA, 1))
    # print(f"{type(X)} {X}")
    
    # Normalize X to help with numerical stability for high-degree polynomials
    X_norm = (X - NORM_VALUE) / float(NORM_VALUE)  # Now values are in [-0.98, 1]
  
    # Create a Vandermonde matrix for polynomial features (columns: x^0, x^1, ..., x^20)
    X_poly = np.vander(X_norm[:, 0], N=DEGREE+1, increasing=True)

    
    # Generate true polynomial coefficients as integers (for degree 20, we need 21 coefficients)
    true_coefs = np.array([0,5,0,-20,0,16])
    # Compute the true y values using the polynomial model: y = sum(true_coefs[i] * (X_norm)^i)
    y_true = X_poly.dot(true_coefs)
    # Add some small noise to simulate measurement error
    noise = randy_savage.random(NUM_OF_DATA) * 0.1
    #print(f"noise \n {type(noise)} {noise.shape} {noise}")
    y = y_true + noise
    
    # Run SGD regression on the polynomial features; learning rate may require tuning.
    theta = sgd_regression(X_poly, y, lr=LEARNING_RATE, epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE)
    
    print("Learned theta:")
    print(theta)
    
    # Print the true polynomial using np.poly1d.
    # np.poly1d expects coefficients in descending order (from x^20 to x^0),
    # so we reverse our true_coefs.
    true_poly = np.poly1d(true_coefs[::-1])
    print("\nTrue polynomial:")
    print(true_poly)

    sorted_idx = np.argsort(X_norm[:, 0])
    X_sorted = X_norm[sorted_idx, 0]

    learned_poly = np.poly1d(theta[::-1])
    true_poly = np.poly1d(true_coefs[::-1])

    y_learned = learned_poly(X_sorted)
    y_true = true_poly(X_sorted)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_norm, y, label="Data", color="grey", alpha=0.5)
    plt.plot(X_sorted, y_true, label="True Polynomial", color="blue", linewidth=2)
    plt.plot(X_sorted, y_learned, label="Learned Polynomial", color="red", linewidth=2)
    plt.xlabel("X (normalized)")
    plt.ylabel("y")
    plt.legend()
    plt.title("Comparison of True and Learned Polynomial")
    plt.show()