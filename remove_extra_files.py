import os

dataset_dirs = ('ADS-B_', 'AST-C_', 'AST-S_', 'Radar_', 'Truth')
cwd = os.getcwd()
data_path = os.path.abspath(os.path.join(cwd, os.pardir, 'data', 'daa-sensors'))

for datadir in dataset_dirs:
    dataset_path = os.path.join(data_path, datadir)
    for fn in os.listdir(dataset_path):
        if datadir == 'Truth':
            if fn.endswith('Tracker.csv'):
                os.remove(os.path.join(dataset_path, fn))
        else:
            if fn.endswith('Truth.csv'):
                os.remove(os.path.join(dataset_path, fn))