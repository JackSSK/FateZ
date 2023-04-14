
import numpy as np
import pandas as pd
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def scale_network(network):
    network = network.replace(np.nan, 0)
    network = np.absolute(network)
    scaled_network = softmax(np.asarray(network))
    return scaled_network