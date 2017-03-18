import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aircraft import Aircraft
from field_mappings import truth_var_map, error_var_map, error_units, truth_units
from tracker_requirements import error_requirements_by_sensor, proportion_tracks_under_error_threshold

OVERWRITE = False
CWD = os.getcwd()
DATAPATH = os.path.abspath(os.path.join(CWD, os.pardir, 'data', 'daa-sensors'))

plt.close('all')


def error_by_var_bin(truth_df, error_df, truth_var, error_var, bin_interval=0.5, valid_filter=False,
                     closing_filter=True, in_volume_filter=False, in_for_filter=False,
                     maneuver_filter=False, print_outliers=False):
    from conversions import feet_to_nmi, rate_of_turn
    from position_calcs import angle_difference

    assert len(truth_df) == len(error_df)

    truth_var_range = {'Horizontal Range': (0, 15),
                       'Tau': (0, 240),
                       'Rate of Turn': (0, 5)}

    keep_bools = np.ones(truth_df.shape[0], dtype=bool)
    if valid_filter:
        keep_bools = error_df['Valid'] == 1
    if closing_filter:
        keep_bools &= ~(np.isnan(truth_df['TauMod']) | np.isinf(
            truth_df['TauMod']))  # only keeping data points when aircraft are closing
    if in_volume_filter:
        keep_bools &= truth_df['InRadarVolume'] == 1
    elif in_for_filter:
        keep_bools &= truth_df['InFOR'] == 1
    if maneuver_filter:
        keep_bools &= truth_df['Roll(Deg)'] == 0
    truth = truth_df[keep_bools]
    error = error_df[keep_bools]
    
    if truth_var == 'Horizontal Range':
        truth['HorizRangeNmi'] = feet_to_nmi(truth['HorizRangeFt'].values)
        bin_var_name = 'HorizRangeNmi'
    elif truth_var == 'Rate of Turn':
        turn_rate = angle_difference(truth['Bearing(Deg)'].values[1:], truth['Bearing(Deg)'].values[:-1])
        turn_rate = np.concatenate((np.ones(1)*turn_rate[0], turn_rate))
        truth['TurnRateDeg'] = np.abs(turn_rate)
        bin_var_name = 'TurnRateDeg'
    else:
        bin_var_name = truth_var_map[truth_var]
    
    bins = np.arange(truth_var_range[truth_var][0], truth_var_range[truth_var][1] + .00000001,
                     bin_interval)
    centers = []
    means = []
    sigmas = []
    percs95list = []
    r95s = []
    outlier_runs = []
    min_val, max_val = (0, 0)
    return_output = {'bins': [], 'means': [], 'percs95': [], 'r95s': []}
    for i in range(len(bins)-1):
        low = bins[i]
        high = bins[i+1]
        center = (high+low)/2
        bin_truth = truth[(truth[bin_var_name] <= high) & (truth[bin_var_name] > low)]
        if len(bin_truth) == 0:
            continue
        if in_volume_filter and (low == 0):
            continue
        bin_error = error.loc[bin_truth.index]
        error_var_name = error_var_map[error_var]
        bin_error_vals = bin_error[error_var_name].values
        mean = bin_error_vals.mean()
        sigma = bin_error_vals.std()
        percs95 = (np.percentile(bin_error_vals, 2.5), np.percentile(bin_error_vals, 97.5))
        r95 = np.percentile(np.abs(bin_error_vals), 95)

        centers.append(center)
        means.append(mean)
        sigmas.append(sigma)
        percs95list.append(percs95)
        r95s.append(r95)
        return_output['bins'].append((high, low))
        return_output['means'].append(mean)
        return_output['percs95'].append(percs95)
        return_output['r95s'].append(r95)

    if print_outliers:
        out_low, out_high = (mean - sigma*15, mean + sigma*15)
        outlier_vals = bin_error[(bin_error[error_var_name] <= out_low) |
                                 (bin_error[error_var_name] >= out_high)]
        outlier_runs.extend(list(outlier_vals.index.levels[0][outlier_vals.index.labels[0]]))
        outlier_runs = np.unique(np.asarray(outlier_runs))
        print(error_var + ' Outlier Runs', len(outlier_runs))
        print(outlier_runs)

    del truth, error

    return return_output


def plot_comparison(set_dirs, truth_var, error_var, label='', legend=False, plot_path=None):
    def get_best_ticks(min_val, max_val):
        steps = (2500, 2000, 1000, 500, 250, 100, 50, 25, 20, 10, 5, 2.5, 1)
        for step in steps:
            ticks = np.arange(min_val, max_val + step, step)
            if len(ticks) > 6:
                return ticks

    def draw_sensor_cutoffs(sensor, error_var):
        try:
            if sensor.startswith('AST'):
                sensor_type = 'AST'
            else:
                sensor_type = sensor
            x_points, y_points = error_requirements_by_sensor[sensor_type].get_plot_params(error_var)
            if not hasattr(y_points, '__iter__'):
                y_points = (y_points, y_points)
            plt.plot(x_points, y_points, color=colors[sensor], lw=2.5, alpha=.75, zorder=2)
            return y_points
        except KeyError:
            return None

    x_units = {'Horizontal Range': 'nmi', 'Tau': 's', 'Rate of Turn': 'deg/s'}
            
    line_formats = {'HALE Correlated': ['solid', 'x'],
                    'HALE Uncorrelated': ['solid', 'o'],
                    'MALE Correlated': ['dashed', 'x'],
                    'MALE Uncorrelated': ['dashed', 'o'],
                    'LEPR Correlated': ['dotted', 'x'],
                    'LEPR Uncorrelated': ['dotted', 'o'],
                    '': ['dashed', 'x']}
    colors = {'Radar': 'r', 'ADS-B': 'b', 'AST-S': 'k', 'AST-C': 'g', 'Radar+ADS-B': 'r',
              'Radar+AST-S': 'k', 'All': 'g'}

    fig = plt.figure(figsize=(9, 6.25))
    ax = fig.gca()
    ax.yaxis.grid(b=True, which='minor', color='.8', linestyle=':')
    ax.yaxis.grid(b=True, which='major', color='.15', linestyle=':')
    ax.xaxis.grid(b=True, which='major', color='.15', linestyle=':')
    max_yval, min_yval = (0, 0)
    max_xval, min_xval = (0, 0)

    sensors = []
    for s in set_dirs:
        truth = pd.read_csv(os.path.join(DATAPATH, s, 'Aggregate_Output', 'truth_all_common_tracker.csv'),
                            index_col=['RunName', 'TimeSecs'])
        error = pd.read_csv(os.path.join(DATAPATH, s, 'Aggregate_Output', 'tracker_all_errors.csv'),
                            index_col=['RunName', 'TimeSecs'])
        if error_var == 'EN Position':
            error['EN Position(ft)'] = np.sqrt(error['East(ft)'].values**2 + error['North(ft)'].values**2)
        sensor, scenario_desc = s.split('_')
        sensors.append(sensor)
        output = error_by_var_bin(truth, error, truth_var, error_var, bin_interval=0.5, closing_filter=False,
                                  valid_filter=True, in_volume_filter=False, in_for_filter=False,
                                  maneuver_filter=False, print_outliers=False)
        x = [(b[0]+b[1])/2 for b in output['bins']]
        r95s = output['r95s']
        if sensor == 'Radar':
            x = x[:-1]
            r95s = r95s[:-1]

        max_xval = max(max_xval, max(x))
        min_xval = min(min_xval, min(x))
        max_yval = max(max_yval, max(r95s))
        min_yval = min(min_yval, min(r95s))

        linestyle, marker = line_formats[scenario_desc]
        plt.plot(x, r95s, color=colors[sensor], linestyle=linestyle, marker=marker, label=sensor+' '+scenario_desc,
                 markerfacecolor='none', markeredgecolor=colors[sensor], markersize=3, linewidth=.5)
        del truth, error

    for sensor in sensors:
        cutoff_y_vals = draw_sensor_cutoffs(sensor, error_var)
        if cutoff_y_vals is not None:
            max_yval = max(max_yval, max(cutoff_y_vals))
            min_yval = min(min_yval, min(cutoff_y_vals))
        
    plt.xlabel(truth_var + ' (' + x_units[truth_var] + ')', fontsize=10)
    plt.ylabel(error_var + ' Error (' + error_units[error_var] + ')', fontsize=10)
    plt.title(error_var + ' Error as a Function of ' + truth_var, fontsize=10)
    plt.xticks(np.arange(0, 16))
    y_ticks_major = get_best_ticks(0, max_yval)
    y_ticks_minor = np.setdiff1d(np.arange(0, y_ticks_major[-1], (y_ticks_major[1]-y_ticks_major[0])/5), y_ticks_major)
    ax.set_yticks(y_ticks_minor, minor=True)
    plt.yticks(y_ticks_major)
    plt.tick_params(axis='both', which='major', labelsize=9)
    plt.xlim(-0.1, 15.1)
    y_vals_range = max_yval - min_yval
    plt.ylim((min_yval-y_vals_range*.025, max_yval+y_vals_range*.05))
    fig.subplots_adjust(bottom=0.2)
    if legend:
        plt.legend(bbox_to_anchor=(.5, -.115), loc='upper center', ncol=4, fontsize=6, handlelength=4)

    if plot_path:
        label_string = label + error_var + '_' + truth_var + '_'
        plt.savefig(os.path.join(plot_path, label_string), dpi=200, bbox_inches='tight')
        plt.close()


alertTimeTargets = {'Corrective': {'early': 75, 'avg': 40, 'late': 20},
                    'Warning': {'early': 55, 'avg': 30, 'late': 15}}


def plot_alerts(truth_df, alert_df, alert_type, plot_path=None, plot_label=''):
    lead_times = []
    run_names = alert_df.index.levels[0]
    for rn in run_names:
        truth_run = truth_df.loc[rn]
        if 1 not in truth_run['HZ'].values:
            continue
        alert_run = alert_df.loc[rn]
        if not alert_run.shape[0] > 0:
            continue
        hz_time = truth_run.index[truth_run['HZ'].values == 1][0]
        try:
            alert_time = alert_run.index[alert_run[alert_type + 'Alert'].values == 1][0]
            lead_times.append(hz_time - alert_time)
        except IndexError:
            lead_times.append(0)
        if lead_times[-1] == 0:
            print(rn)

    fig = plt.figure()
    ax = fig.gca()
    plt.hist(lead_times, bins=50)
    mean = np.array(lead_times).mean()
    plt.axvline(mean, linewidth=3, color='k')
    plt.axvline(alertTimeTargets[alert_type]['avg'], linewidth=2, linestyle='dashed', color='.5')
    plt.axvline(alertTimeTargets[alert_type]['early'], linewidth=1, linestyle='dashed', color='r')
    plt.axvline(alertTimeTargets[alert_type]['late'], linewidth=1, linestyle='dashed', color='r')
    plt.xlabel(alert_type + ' Alert HZ Lead Time (s)')
    plt.ylabel('# of Scenarios')

    if plot_path:
        plt.savefig(os.path.join(plot_path, alert_type + 'Alerts' + plot_label), dpi=400)


def plot_sensor_errors(sensor_df, sensor_type, plot_path=None):
    from conversions import knots_to_mps, fpm_to_mps
    units = {'Azimuth': 'deg', 'Range': 'ft', 'Range Rate': 'ft/sec',
             'Elevation': 'deg', 'Latitude': 'secs', 'Longitude': 'secs',
             'Altitude': 'ft', 'East Velocity': 'm/s', 'North Velocity': 'm/s',
             'Up Velocity': 'm/s', 'Yaw': 'deg', 'Pitch': 'deg', 'Roll': 'deg'}
    field_map = {'Azimuth': 'Azimuth', 'Range': 'Range', 'Elevation': 'Elevation',
                 'Latitude': 'Latitude(deg)', 'Longitude': 'Longitude(deg)',
                 'Altitude': 'Altitude(ft)', 'East Velocity': 'EastVel(kts)',
                 'North Velocity': 'NorthVel(kts)', 'Up Velocity': 'UpVel(fpm)',
                 'Yaw': 'Yaw(deg)', 'Pitch': 'Pitch(deg)', 'Roll': 'Roll(deg)'}
    sensor_fields = {'Radar': ('Azimuth', 'Range', 'Elevation'),
                     'AST-S': ('Azimuth', 'Range'),
                     'ADS-B': ('Latitude', 'Longitude', 'Altitude',
                               'East Velocity', 'North Velocity', 'Up Velocity',
                               'Yaw', 'Pitch', 'Roll')}
    for field in sensor_fields[sensor_type]:
        fig = plt.figure()
        ax = fig.gca()
        vals = sensor_df[field_map[field]].values
        if (field == 'Latitude') or (field == 'Longitude'):
            vals *= 3600
        elif (field == 'East Velocity') or (field == 'North Velocity'):
            vals = knots_to_mps(vals)
        elif field == 'Up Velocity':
            vals = fpm_to_mps(vals)
        percs95 = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
        heights, _, _ = plt.hist(vals, bins=100)
        plt.axvline(percs95[0], linewidth=3, color='k')
        plt.axvline(percs95[1], linewidth=3, color='k')

        plt.xlabel(field+' Error ('+units[field]+')')
        plt.ylabel('Frequency')
        plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        plt.title('Distribution of Raw ' + sensor_type + ' ' + field + ' Errors')
        if plot_path:
            plt.savefig(os.path.join(plot_path, sensor_type + '_' + field + '_Errors_Hist'), dpi=300)


def write_all_data(scenario_set_directory, valid_wait_secs=0):
    from load_data import load_tracker_error_settled
    scenario_set_path = os.path.join(DATAPATH, scenario_set_directory, 'Scenario_Output')
    truth_path = os.path.join(scenario_set_path, 'Truth', 'Truth_Logs')
    adsb_error_path = os.path.join(scenario_set_path, 'ADS-B', 'Error_Logs')
    radar_error_path = os.path.join(scenario_set_path, 'Radar', 'Error_Logs')
    tracker_error_path = os.path.join(scenario_set_path, 'Tracker', 'Error_Logs')
    output_path = os.path.join(DATAPATH, scenario_set_directory, 'Aggregate_Output')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    truth_all_common_tracker = pd.DataFrame()
    truth_all_common_radar = pd.DataFrame()
    adsb_all_errors_ownship = pd.DataFrame()
    adsb_all_errors_intruder = pd.DataFrame()
    radar_all_errors = pd.DataFrame()
    tracker_all_errors = pd.DataFrame()
    set_summary = {'NumRuns': 0, 'InFOR': 0, 'InRadarVolume': 0, 'NHZ': 0, 'MZ': 0, 'HZ': 0,
                   'MissedAlerts': 0, 'NoTrackerData': 0}
    for run_filename in os.listdir(truth_path):
        run_name = run_filename.rstrip('_intruder_truth.csv')
        print(run_name)
        radar_error_data = None
        tracker_error_data = None
        truth_data = pd.read_csv(os.path.join(truth_path, run_filename), index_col=0)
        truth_data['RunName'] = run_name

        set_summary['NumRuns'] += 1
        if 1 in truth_data['InFOR'].values:
            set_summary['InFOR'] += 1
        if 1 in truth_data['InRadarVolume'].values:
            set_summary['InRadarVolume'] += 1
        if 1 in truth_data['HZ'].values:
            set_summary['HZ'] += 1
        elif 1 in truth_data['MZ'].values:
            set_summary['MZ'] += 1
        else:
            set_summary['NHZ'] += 1

        try:
            adsb_error_data = pd.read_csv(os.path.join(adsb_error_path, run_name+'_adsb.csv'), index_col=[0, 1])
            adsb_error_data['RunName'] = run_name
            adsb_error_own = adsb_error_data.loc[1].iloc[10:]
            adsb_all_errors_ownship = adsb_all_errors_ownship.append(adsb_error_own)
            adsb_error_int = adsb_error_data.loc[2].iloc[10:]
            adsb_all_errors_intruder = adsb_all_errors_intruder.append(adsb_error_int)
        except OSError:
            pass
        try:
            radar_error_data = pd.read_csv(os.path.join(radar_error_path, run_name+'_radar.csv'), index_col=0).iloc[1:]
            radar_error_data['RunName'] = run_name
        except OSError:
            pass
        try:
            tracker_error_data = load_tracker_error_settled(os.path.join(tracker_error_path, run_name+'_tracker.csv'),
                                                            valid_wait_secs)
            tracker_error_data['RunName'] = run_name
        except OSError:
            pass

        if (type(radar_error_data) != type(None)) or (type(tracker_error_data) != type(None)):
            intruder = Aircraft(truth_data, radar_error_data, tracker_error_data)

        if type(radar_error_data) != type(None):
            truth_interp = intruder.get_truth_interpolated(radar_error_data, True)
            truth_interp['RunName'] = run_name
            truth_all_common_radar = truth_all_common_radar.append(truth_interp)
            radar_all_errors = radar_all_errors.append(radar_error_data)
        
        if type(tracker_error_data) != type(None):
            truth_interp = intruder.get_truth_interpolated(tracker_error_data, True)
            truth_interp['RunName'] = run_name
            truth_all_common_tracker = truth_all_common_tracker.append(truth_interp)
            tracker_all_errors = tracker_all_errors.append(tracker_error_data)
        else:
            set_summary['NoTrackerData'] += 1
            print('No Tracker Data')
            
    adsb_all_errors_ownship.to_csv(os.path.join(output_path, 'adsb_all_errors_ownship.csv'))
    adsb_all_errors_intruder.to_csv(os.path.join(output_path, 'adsb_all_errors_intruder.csv'))
    radar_all_errors.to_csv(os.path.join(output_path, 'radar_all_errors.csv'))
    if len(truth_all_common_tracker) > 0:
        truth_all_common_tracker.reset_index(inplace=True)
        truth_all_common_tracker.set_index(['RunName', 'TimeSecs'], inplace=True)
        for zone in ('HZ', 'MZ', 'NHZ'):
            truth_all_common_tracker[zone] = np.round(truth_all_common_tracker[zone].values)
        truth_all_common_tracker.to_csv(os.path.join(output_path, 'truth_all_common_tracker.csv'))

        tracker_all_errors.reset_index(inplace=True)
        tracker_all_errors.set_index(['RunName', 'TimeSecs'], inplace=True)
        tracker_all_errors.to_csv(os.path.join(output_path, 'tracker_all_errors.csv'))

    pd.DataFrame(set_summary, index=[0]).to_csv(os.path.join(output_path, 'set_summary.csv'), index=False)


def write_scenario_set_output(scenario_set_directory, valid_wait_secs=0):
    if (not os.path.exists(os.path.join(DATAPATH, scenario_set_directory, 'Aggregate_Output', 'Tracker'))) or \
            OVERWRITE:
        output_path = os.path.join(DATAPATH, scenario_set_directory, 'Aggregate_Output')
        adsb_output_path = os.path.join(output_path, 'ADS-B')
        adsb_ownship_output_path = os.path.join(adsb_output_path, 'Ownship')
        adsb_intruder_output_path = os.path.join(adsb_output_path, 'Intruder')
        radar_output_path = os.path.join(output_path, 'Radar')
        tracker_output_path = os.path.join(output_path, 'Tracker')

        for p in (output_path, adsb_output_path, adsb_ownship_output_path, adsb_intruder_output_path,
                  radar_output_path, tracker_output_path):
            if not os.path.exists(p):
                os.mkdir(p)

        write_all_data(scenario_set_directory, valid_wait_secs)


def write_tracker_requirements_by_scenario(scenario_set_directory):
    from conversions import feet_to_nmi
    write_scenario_set_output(scenario_set_directory)
    sensor = scenario_set_directory.split('_')[0]
    if sensor.startswith('AST'):
        sensor = 'AST'
    log = {'ScenarioID': [], 'ProportionMet': [], 'ProportionValidMet': []}
    for fn in os.listdir(os.path.join(DATAPATH, scenario_set_directory, 'Scenario_Output', 'Truth',
                                      'Truth_Logs')):
        scenario_name = fn.rsplit('_', 2)[0]
        print(scenario_name)
        scenario_id = scenario_name.split('_')[0]
        log['ScenarioID'].append(scenario_id)
        truth_df_tmp = pd.read_csv(os.path.join(DATAPATH, scenario_set_directory, 'Scenario_Output', 'Truth',
                                                'Truth_Logs', fn))
        if truth_units['Horizontal Range'] == 'ft':
            in_valid_range = error_requirements_by_sensor[sensor].in_valid_range(
                feet_to_nmi(truth_df_tmp[truth_var_map['Horizontal Range']]))
        else:
            in_valid_range = error_requirements_by_sensor[sensor].in_valid_range(
                truth_df_tmp[truth_var_map['Horizontal Range']])
        if sum(in_valid_range) == 0:
            log['ProportionMet'].append('TruthOutsideBounds')
            log ['ProportionValidMet'].append('TruthOutsideBounds')
            continue
        try:
            error_df = pd.read_csv(os.path.join(DATAPATH, scenario_set_directory, 'Scenario_Output', 'Tracker',
                                                'Error_Logs', scenario_name + '_tracker.csv'))
            truth_df_tmp.set_index('TimeSecs', inplace=True)
            error_df.set_index('TimeSecs', inplace=True)
            truth_df = Aircraft(truth_df_tmp).get_truth_interpolated(error_df, True)
            proportion_met = proportion_tracks_under_error_threshold(truth_df, error_df, sensor)
            log['ProportionMet'].append(proportion_met)
            proportion_valid_met = proportion_tracks_under_error_threshold(truth_df, error_df, sensor, valid_filter=True)
            log['ProportionValidMet'].append(proportion_valid_met)
        except OSError:  # no tracker data
            log['ProportionMet'].append('NoTrackerData')
            log['ProportionValidMet'].append('NoTrackerData')
    pd.DataFrame(log).to_csv(os.path.join(DATAPATH, scenario_set_directory, 'Aggregate_Output', 'Tracker',
                                          'Error_requirements_log.csv'))


if __name__ == '__main__':
    scenario_sets = ['ADS-B_', 'Radar_', 'AST-S_', 'AST-C_']

    for s_set in scenario_sets:
        write_scenario_set_output(s_set)
        # write_tracker_requirements_by_scenario(s_set)

    comparison_output_path = os.path.join(DATAPATH, 'Comparisons')
    if not os.path.exists(comparison_output_path):
        os.mkdir(comparison_output_path)

    for error_var in ['Relative Altitude', 'Ground Track',
                      'Ground Speed', 'Vertical Rate', 'Horizontal Position',
                      'Relative Velocity']:
        plot_comparison(scenario_sets, 'Horizontal Range', error_var, plot_path=comparison_output_path)
