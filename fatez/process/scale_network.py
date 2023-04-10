from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def scale_network(network):
    network = network.replace(np.nan, 0)
    network = np.absolute(network)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_network = scaler.fit_transform(np.asarray(network))
    return scaled_network