from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

def normalization(X,Y):
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    Y = Y[:X.shape[0], 0] # Selects Machine Failure as our predictor
    x = scaler_X.fit_transform(X)
    y = scaler_Y.fit_transform(Y.reshape(-1, 1))
    return  x, y# X_norm


def get_ai4i_dataset():
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)
    X_df = ai4i_2020_predictive_maintenance_dataset.data.features.copy()
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    X_df['Type'] = X_df['Type'].map(type_mapping)
    X = X_df.to_numpy()
    if X.ndim == 1:
        X = X.reshape(-1, len(X_df.columns))  # Reshape to (num_samples, num_features)
    # Extract targets
    y = ai4i_2020_predictive_maintenance_dataset.data.targets.to_numpy()
    print(f"X shape: {X.shape}, y shape: {y.shape}")  # Debugging statement
    X_norm, y_norm = normalization(X,y)
    return X_norm, y_norm
