import numpy as np
from to_Latex import to_latex
from energydata_predictions import read_energy_dataset
from get_ai4i2020_data import get_ai4i_dataset
from ccpp import get_ccpp_data
from sklearn.linear_model import LinearRegression


def main():
    print("Energy Data:")
    X_e, y_e = read_energy_dataset()
    model = LinearRegression()
    model.fit(X_e, y_e)
    theta_values = np.append(model.intercept_, model.coef_)
    print(to_latex(theta_values))
    print("\n\n")
    print("Combined Cycle Power Plant Data:")
    X_c, y_c = get_ccpp_data()
    model = LinearRegression()
    model.fit(X_c, y_c)
    theta_values = np.append(model.intercept_, model.coef_)
    print(to_latex(theta_values))
    print("\n\n")
    print("AI4I2020 Data:")
    X_a, y_a = get_ai4i_dataset()
    model = LinearRegression()
    model.fit(X_a, y_a)
    theta_values = np.append(model.intercept_, model.coef_)
    print(to_latex(theta_values))

if __name__ == "__main__":
    main()