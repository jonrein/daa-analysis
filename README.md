These scripts process truth, sensor, and tracker data logs from DAA scenarios.

It is assumed that there is a data directory, with a "Truth" subdirectory and additional subdirectories for each sensor configuration of interest.  It is further assumed that the sensor configuration subdirectories have a unique directory for each scenario run, which contains all of the relevant data logs (other than truth) for that run.  If there is no such directory structure, run make_scenario_dirs.py.

analyze_scenarios.py generates scenario-level data and plots for a specified sensor configuration directory

analyze_aggerate.py generates aggregate data and plots for the specified sensor configuration directory(ies)
