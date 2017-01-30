import os
from aircraft import Aircraft
from own_int_pair import OwnIntPair
from load_data import load_truth, load_radar, load_tracker, load_cat21

SCENARIO_DIR = 'ADS-B_'
LOG_CAT21 = False

cwd = os.getcwd()
data_path = os.path.abspath(os.path.join(cwd, os.pardir, 'data', 'daa-sensors'))
truth_path = os.path.join(data_path, 'Truth')
runs_path = os.path.join(data_path, SCENARIO_DIR, 'Runs')
output_path = os.path.join(data_path, SCENARIO_DIR, 'Scenario_Output')
adsb_output_path = os.path.join(output_path, 'ADS-B')
radar_output_path = os.path.join(output_path, 'Radar')
tracker_output_path = os.path.join(output_path, 'Tracker')
truth_output_path = os.path.join(output_path, 'Truth')
adsb_error_log_path = os.path.join(adsb_output_path, 'Error_Logs')
radar_error_plot_path = os.path.join(radar_output_path, 'Error_Plots')
radar_value_plot_path = os.path.join(radar_output_path, 'Value_Plots')
radar_error_log_path = os.path.join(radar_output_path, 'Error_Logs')
tracker_alert_log_path = os.path.join(tracker_output_path, 'Alert_Logs')
tracker_error_plot_path = os.path.join(tracker_output_path, 'Error_Plots')
tracker_value_plot_path = os.path.join(tracker_output_path, 'Value_Plots')
tracker_error_log_path = os.path.join(tracker_output_path, 'Error_Logs')
truth_pos_plot_path = os.path.join(truth_output_path, 'Position_Plots')
truth_log_path = os.path.join(truth_output_path, 'Truth_Logs')
for p in (output_path, adsb_output_path, radar_output_path, tracker_output_path, truth_output_path,
          adsb_error_log_path, radar_error_plot_path, radar_value_plot_path, radar_error_log_path,
          tracker_alert_log_path, tracker_error_plot_path, tracker_value_plot_path,
          tracker_error_log_path, truth_pos_plot_path, truth_log_path):
    if not os.path.exists(p):
        os.mkdir(p)

truth_files = dict(zip([fn.split('_', 1)[0] for fn in os.listdir(truth_path)],
                       [os.path.join(truth_path, fn) for fn in os.listdir(truth_path)]))

for i, run_dir in enumerate(os.listdir(runs_path)):
    print(i, run_dir)
    adsb_data = {1: None, 2: None}
    radar_data = None
    tracker_data = None
    for run_file in os.listdir(os.path.join(runs_path, run_dir)):
        if not run_file.endswith('.csv'):
            continue
        run_file_path = os.path.join(runs_path, run_dir, run_file)
        if run_file.endswith('CAT48.csv'):
            radar_data = load_radar(run_file_path)
        elif run_file.endswith('CAT21.csv') and LOG_CAT21:
            adsb_data_tmp = load_cat21(run_file_path)
            if len(adsb_data_tmp) > 0:
                adsb_data = adsb_data_tmp
        elif run_file.endswith('Tracker.csv'):
            tracker_data = load_tracker(run_file_path)
        run_label = run_file.rsplit('_', 1)[0]
        scenario_id = run_label.split('_', 1)[0]

    truth_data = load_truth(truth_files[scenario_id])

    if type(tracker_data) != type(None):
        ownship = Aircraft(truth_data[1], radar_data, adsb_data[1], tracker_data[1])
        intruder = Aircraft(truth_data[2], radar_data, adsb_data[2], tracker_data[2])
    else:
        ownship = Aircraft(truth_data[1], radar_data, adsb_data[1], None)
        intruder = Aircraft(truth_data[2], radar_data, adsb_data[2], None)
    pair = OwnIntPair(ownship, intruder)

    pair.plot_positions(run_dir + '_tracks', truth_pos_plot_path)
    pair.intruder.truth_df.to_csv(os.path.join(truth_log_path, run_dir + '_intruder_truth.csv'))
    try:
        pair.adsb_report(run_label + '_adsb', adsb_error_log_path)
    except OwnIntPair.NoAdsbDataException:
        pass
    try:
        pair.radar_report(run_label + '_radar', radar_error_log_path)
    except OwnIntPair.NoRadarDataException:
        pass
    try:
        pair.tracker_report(run_label + '_tracker', tracker_alert_log_path, tracker_error_log_path)
    except OwnIntPair.NoTrackerDataException:
        pass
