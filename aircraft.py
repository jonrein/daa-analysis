import pandas as pd

BASE_RDR = {'small': 5.4, 'medium': 6, 'large': 6.7}  # nmi
RDR_AIRSPEED_CUTOFFS = (100, 130, 1000)  # kts


def get_size(airspeed):
    if airspeed < RDR_AIRSPEED_CUTOFFS[0]:
        return 'small'
    elif airspeed < RDR_AIRSPEED_CUTOFFS[1]:
        return 'medium'
    else:
        return 'large'


def get_base_rdr(size):
    return BASE_RDR[size]


class Aircraft(object):
    def __init__(self, truth_df, radar_df=None, adsb_df=None, tracker_df=None):
        self.truth_df = truth_df
        self.airspeed = self.truth_df['TAirSpd(knots)'].max()
        self.size = get_size(self.airspeed)
        self.base_rdr = get_base_rdr(self.size)
        self.adsb_df = None
        self.radar_df = None
        self.tracker_df = None
        self.truth_interp_self_adsb_df = None
        self.truth_interp_radar_df = None
        self.truth_interp_tracker_df = None

        if type(radar_df) != type(None):
            self.radar_df = radar_df
        if type(adsb_df) != type(None):
            self.adsb_df = adsb_df
        if type(tracker_df) != type(None):
            self.tracker_df = tracker_df

    def get_truth_interpolated(self, reference_df, kept_fields=('Alt(ft)', 'LatD', 'LonD')):
        """referenceDF: DataFrame with indexed times that are being interpolated on
           keptFields: list/tuple of str field names to keep, or True to keep all"""
        import numpy as np
        if kept_fields:
            truth_df = self.truth_df.copy()
            kept_fields = truth_df.columns
        else:
            truth_df = self.truth_df[kept_fields]
        common_times = np.intersect1d(truth_df.index, reference_df.index)
        ref_unique_times = np.unique(np.array([i for i in reference_df.index if i not in common_times]))
        empty_df = pd.DataFrame(index=ref_unique_times, columns=kept_fields)
        truth_df = pd.concat([truth_df, empty_df])
        for field in kept_fields:
            truth_df[field].interpolate(method='values', inplace=True)
        return truth_df.loc[reference_df.index]
