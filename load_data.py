import numpy as np
import pandas as pd
from conversions import geod_2_enu_pos


def find_column_indexes(data_path, field_list):
    import csv
    with open(data_path, 'r') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        headers = dict_reader.fieldnames
    return [headers.index(f) for f in field_list if f in headers]


def load_truth(data_path, own_mode_3a=1):
    keep_cols = ('ScenarioTime', 'M3A', 'Lat(Deg)', 'Lon(Deg)', 'Alt(ft)', 'GrndSpd(knots)',
                 'TAirSpd(knots)', 'Bearing(Deg)', 'NorthVel(knots)', 'EastVel(knots)',
                 'DownVel(FPM)', 'Pitch(Deg)', 'Roll(Deg)')
    data = pd.read_csv(data_path, usecols=find_column_indexes(data_path, keep_cols))
    beacons = data['M3A'].unique()
    data.rename(columns={'ScenarioTime': 'TimeSecs', 'Lat(Deg)': 'LatD',
                         'Lon(Deg)': 'LonD'}, inplace=True)

    data.set_index(['M3A', 'TimeSecs'], inplace=True)

    latlon_center = (np.array([data['LatD'].min(), data['LatD'].max()]).mean(),
                     np.array([data['LonD'].min(), data['LonD'].max()]).mean(),
                     np.array([data['Alt(ft)'].min(), data['Alt(ft)'].max()]).mean())

    e, n, _ = geod_2_enu_pos((data['LatD'].values, data['LonD'].values, data['Alt(ft)'].values),
                             latlon_center)
    data['X'] = e
    data['Y'] = n

    own_first = data.loc[own_mode_3a].iloc[0]
    first_y = own_first['Y']
    first_x = own_first['X']
    data['Y'] = data['Y'] - first_y
    data['X'] = data['X'] - first_x

    data_dict = {}
    for b in beacons:
        data_dict[b] = data.loc[b]

    return data_dict


def load_cat21(data_path, own_mode_3a=1):
    data = pd.read_csv(data_path)
    beacons = data['M3A'].unique()
    data.rename(columns={'ScenarioTime': 'TimeSecs'}, inplace=True)
    data.set_index(['M3A', 'TimeSecs'], inplace=True)
    data_dict = {}
    for b in beacons:
        if b == own_mode_3a:
            data_dict[b] = data.loc[b].iloc[::20]
        else:
            data_dict[b] = data.loc[b]
    return data_dict


def load_radar(data_path):
    data = pd.read_csv(data_path)
    data.rename(columns={'ScenarioTime': 'TimeSecs'}, inplace=True)
    data.set_index('TimeSecs', inplace=True)
    return data


def load_tracker(data_path):
    keep_cols = ('ScenarioTime', 'M3A', 'TrkNum', 'Lat(Deg)', 'Lon(Deg)', 'BaroAlt(ft)',
                 'EDot(nmi/s)', 'NDot(nmi/s)', 'AltRate(ft/min)',
                 'EStdDev(m)', 'NStdDev(m)', 'ENStdDev(m)', 'EDotStdDev(nmi/s)', 'NDotStdDev(nmi/s)',
                 'BaroStdDev(ft)', 'AltRateStdDev(ft/min)', 'HPosValid', 'VPosValid',
                 'HVelValid', 'VVelValid')
    data = pd.read_csv(data_path, usecols=find_column_indexes(data_path, keep_cols))
                
    beacons = data['M3A'].unique()
    data.rename(columns={'ScenarioTime': 'TimeSecs'}, inplace=True)
    data['TimeSecs'] = np.around(data['TimeSecs'].values)
    data.set_index(['TimeSecs'], inplace=True)

    data_dict = {1: data[data['M3A'] == 1],
                 2: data[data['M3A'] != 1]}
#    for b in beacons:
#        data_dict[b] = data.loc[b]
                
    return data_dict


def load_tracker_error_settled(data_path, n_secs):
    data_all = pd.read_csv(data_path)
    track_ids = data_all['TrackID'].unique()
    data = pd.DataFrame()
    for track_id in track_ids:
        data_track = data_all[data_all['TrackID'] == track_id]
        first_sec = data_track.iloc[0]['TimeSecs']
        data_track_settled = data_track[data_track['TimeSecs'] >= (first_sec + n_secs)]
        data = data.append(data_track_settled)
    data.set_index(['TimeSecs'], inplace=True)
    return data
