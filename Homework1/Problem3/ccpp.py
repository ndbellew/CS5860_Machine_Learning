from ucimlrepo import fetch_ucirepo

# metadata
# print(combined_cycle_power_plant.metadata)

# variable information
# print(combined_cycle_power_plant.variables)
def get_ccpp_data():
    combined_cycle_power_plant = fetch_ucirepo(id=294)
    X = combined_cycle_power_plant.data.features.to_numpy()
    y = combined_cycle_power_plant.data.targets.to_numpy()
    return X, y