from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler


def normalization(X,Y):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    x = scaler_X.fit_transform(X)
    y = scaler_Y.fit_transform(Y.reshape(-1, 1)).flatten()
    return  x, y# X_norm

# metadata
# print(combined_cycle_power_plant.metadata)

# variable information
# print(combined_cycle_power_plant.variables)
def get_ccpp_data():
    combined_cycle_power_plant = fetch_ucirepo(id=294)
    X = combined_cycle_power_plant.data.features.to_numpy()
    y = combined_cycle_power_plant.data.targets.to_numpy()
    X_norm, y_norm = normalization(X, y)
    return X_norm, y_norm