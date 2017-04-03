import numpy as np
from field_mappings import truth_var_map, error_var_map, truth_units
from conversions import feet_to_nmi


class SensorError(object):
    def __init__(self, thresholds, valid_h_range):
        self.thresholds = thresholds
        self.h_range_low = valid_h_range[0]
        self.h_range_high = valid_h_range[1]

    def in_valid_range(self, h_range):
        return (h_range >= self.h_range_low) & (h_range <= self.h_range_high)

    def get_threshold_vals(self, h_range, error_var):
        threshold = self.thresholds[error_var]
        if type(threshold) == int:
            threshold_val_func = lambda x: threshold
        else:
            slope, intercept = threshold
            threshold_val_func = lambda x: slope*(x - self.h_range_low) + intercept
        return threshold_val_func(h_range)

    def at_or_below_threshold(self, h_range, val, error_var):
        return val <= self.get_threshold_vals(h_range, error_var)

    def get_plot_params(self, error_var):
        range_ends = [self.h_range_low, self.h_range_high]
        return range_ends, self.get_threshold_vals(np.array(range_ends), error_var)

error_requirements_by_sensor = {'ADS-B': SensorError({'Horizontal Position': 900,
                                                      'Relative Altitude': 300,
                                                      'Relative Velocity': 30,
                                                      'Vertical Rate': 400},
                                                     (0, 20)),
                                'AST': SensorError({'Horizontal Position': (1000, 1250),
                                                    'Relative Altitude': 300,
                                                    'Relative Velocity': (33, 85),
                                                    'Vertical Rate': 400},
                                                   (0.5, 14)),
                                'Radar': SensorError({'Horizontal Position': (125, 250),
                                                      'Relative Altitude': (100, 150),
                                                      'Relative Velocity': (10, 50),
                                                      'Vertical Rate': (380, 800)},
                                                     (1, 6.7))}


def first_good_track_time(truth_df, error_df, sensor):
    assert len(truth_df) == len(error_df)
    # Filter data
    keep_bools = np.ones(truth_df.shape[0], dtype=bool)
    keep_bools &= error_requirements_by_sensor[sensor].in_valid_range(
        feet_to_nmi(truth_df[truth_var_map['Horizontal Range']]))
    keep_bools = error_df['Valid'] == 1
    try:
        first_ix = np.nonzero(keep_bools)[0][0]
        return truth_df.index[first_ix]
    except IndexError:
        return None


def proportion_tracks_under_error_threshold(truth_df, error_df, sensor, error_var='all', require_valid_flag=True,
                                            detection_range_filter=True, valid_filter=False):
    assert len(truth_df) == len(error_df)
    assert error_var in ('Horizontal Position', 'Relative Altitude', 'Relative Velocity', 'Vertical Rate', 'all')
    # Filter data
    keep_bools = np.ones(truth_df.shape[0], dtype=bool)
    if detection_range_filter:
        if truth_units['Horizontal Range'] == 'ft':
            keep_bools &= error_requirements_by_sensor[sensor].in_valid_range(
                feet_to_nmi(truth_df[truth_var_map['Horizontal Range']]))
        else:
            keep_bools &= error_requirements_by_sensor[sensor].in_valid_range(
                truth_df[truth_var_map['Horizontal Range']])
    if valid_filter:
        keep_bools = error_df['Valid'] == 1
    if sum(keep_bools) == 0:
        return 'NoTrackerDataWithinBounds'
    truth_filt = truth_df[keep_bools]
    error_filt = error_df[keep_bools]

    # Calculate proportion
    if error_var == 'all':
        under_bools = np.ones(truth_filt.shape[0], dtype=bool)
        for ev in ('Horizontal Position', 'Relative Altitude', 'Relative Velocity', 'Vertical Rate'):
            under_bools &= error_requirements_by_sensor[sensor].at_or_below_threshold(
                truth_filt[truth_var_map['Horizontal Range']], np.abs(error_filt[error_var_map[ev]]), ev)
    else:
        under_bools = error_requirements_by_sensor[sensor].at_or_below_threshold(
            truth_filt[truth_var_map['Horizontal Range']], np.abs(error_filt[error_var_map[error_var]]), error_var)
    if require_valid_flag and not valid_filter:
        under_bools &= error_filt['Valid'] == 1
    return sum(under_bools) / truth_filt.shape[0]