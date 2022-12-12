import pandas as pd
from pkg_resources import resource_filename

# data_path is based on current dir. Edit it here
data_path = '../data/mouse/Transfac.....txt'
path = resource_filename(__name__, data_path)
data = pd.read_csv(path, sep = '\t')
