import numpy as np
import matplotlib.pyplot as plt
# Create an array of x values
x = np.linspace(-1, 1, 400)

# Compute T₅(x)
y = 16*x**5 - 20*x**3 + 5*x

# Plot the function
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("T₅(x)")
plt.title("Chebyshev Polynomial T₅(x)")
plt.grid(True)
plt.show()