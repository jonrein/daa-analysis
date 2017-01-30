import os
from shutil import move

DIRECTORIES = ['AST-C_', 'ADS-B_', 'AST-S_', 'Radar_']
               
cwd = os.getcwd()
data_path = os.path.abspath(os.path.join(cwd, os.pardir, 'data', 'daa-sensors'))

for d in DIRECTORIES:
    print(d)
    runs_path = os.path.join(data_path, d)
    file_names = os.listdir(runs_path)
    new_runs_path = os.path.join(runs_path, 'Runs')
    os.mkdir(new_runs_path)
    scenario_names = []
    for file_name in file_names:
        if file_name.endswith('Tracker.csv'):
            scenario_name = file_name.rsplit('_', 1)[0]
            os.mkdir(os.path.join(new_runs_path, scenario_name))

    for file_name in file_names:
        scenario_name = file_name.rsplit('_', 1)[0]
        old_path = os.path.join(runs_path, file_name)
        new_path = os.path.join(new_runs_path, scenario_name, file_name)
        move(old_path, new_path)
