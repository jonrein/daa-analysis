import numpy as np
import matplotlib.pyplot as plt
from conversions import feet_to_nmi, nmi_to_feet

BEARING_CUTOFFS = (30, 60, 90, 181)  # degrees
CORRECTION_FACTORS = {'small': (1, 0.84, 0.46, 0.45),
                      'medium': (1, 0.84, 0.46, 0.45),
                      'large': (1, 0.84, 0.46, 0.45)}
RCPR = 0  # nmi
RADAR_FOR_AZIMUTH = 110  # +/- 110 degrees horizontal
RADAR_FOR_VERTICAL = 15  # +/- 15 degrees vertical


class OwnIntPair(object):
    def __init__(self, ownship_aircraft, intruder_aircraft):
        self.ownship = ownship_aircraft
        self.intruder = intruder_aircraft
        if self.ownship.truth_df.shape[0] > self.intruder.truth_df.shape[0]:
            self.ownship.truth_df = self.ownship.truth_df.loc[self.intruder.truth_df.index]

        self.set_true_relative_position()
        if type(self.intruder.radar_df) != type(None):
            self.set_interp_df_radar()
        if type(self.intruder.adsb_df) != type(None):
            self.set_interp_df_adsb()
        if type(self.intruder.tracker_df) != type(None):
            self.set_interp_df_tracker()

    def set_true_relative_position(self):
        from conversions import geod_2_enu_pos, angle_360_to_180, knots_to_fps
        from position_calcs import bearing_from_en_coords, hazard_zone_classification
        from coord_transforms import rotate_3d

        own_df = self.ownship.truth_df
        int_df = self.intruder.truth_df

        int_enu = geod_2_enu_pos((int_df['LatD'].values, int_df['LonD'].values, int_df['Alt(ft)'].values),
                                 (own_df['LatD'].values, own_df['LonD'].values, own_df['Alt(ft)'].values), 'ft')
        int_df['EFt'] = int_enu[0]
        int_df['NFt'] = int_enu[1]
        int_df['UFt'] = int_enu[2]
        int_df['RangeFt'] = np.sqrt(int_enu[0] ** 2 + int_enu[1] ** 2 + int_enu[2] ** 2)
        int_df['AzimuthBFUncorrectedDeg'] = bearing_from_en_coords(int_enu[:2])
        time_diffs = int_df.index.values[10:] - int_df.index.values[:-10]
        range_rt = (int_df['RangeFt'].values[10:] - int_df['RangeFt'].values[:-10]) / time_diffs
        range_rt = np.hstack((np.ones(10) * range_rt[0], range_rt))
        int_df['RangeRtFtSec'] = range_rt
        int_df['HorizRangeFt'] = np.sqrt(int_enu[0] ** 2 + int_enu[1] ** 2)
        int_df['HorizRangeRtFtSec'] =  np.sqrt((int_enu[1] + knots_to_fps(int_df['NorthVel(knots)'].values -
                                                                 own_df['NorthVel(knots)'].values))**2 +
                                              (int_enu[0] + knots_to_fps(int_df['EastVel(knots)'].values -
                                                                 own_df['EastVel(knots)'].values))**2) - \
                                       int_df['HorizRangeFt'].values

        int_df['RelAltFt'] = int_df['Alt(ft)'].values - own_df['Alt(ft)'].values

        azimuth_bodyframe = np.zeros(int_df.shape[0])
        elevation_bodyframe = np.zeros(int_df.shape[0])
        in_for = np.zeros(int_df.shape[0])
        in_volume = np.zeros(int_df.shape[0])
        hz = np.zeros(int_df.shape[0])
        mz = np.zeros(int_df.shape[0])
        nhz = np.zeros(int_df.shape[0])
        tau_mod = np.zeros(int_df.shape[0])
        hmd = np.zeros(int_df.shape[0])
        vmd = np.zeros(int_df.shape[0])
        corrective_alert = np.zeros(int_df.shape[0])
        warning_alert = np.zeros(int_df.shape[0])
        for i in range(int_df.shape[0]):
            own_i = own_df.iloc[i]
            int_i = int_df.iloc[i]
            own_yaw, own_pitch, own_roll = (angle_360_to_180(own_df.iloc[i]['Bearing(Deg)']),
                                            own_df.iloc[i]['Pitch(Deg)'], own_df.iloc[i]['Roll(Deg)'])
            enu_bodyframe = rotate_3d((int_df.iloc[i]['EFt'], int_df.iloc[i]['NFt'], int_df.iloc[i]['UFt']),
                                      own_yaw, own_pitch, own_roll)
            range_nm = feet_to_nmi(int_i['RangeFt'])
            azimuth_bodyframe[i] = bearing_from_en_coords(enu_bodyframe[:2])
            elevation_bodyframe[i] = np.rad2deg(np.arcsin(enu_bodyframe[2] / int_i['RangeFt']))
            if (np.abs(angle_360_to_180(azimuth_bodyframe[i])) <= RADAR_FOR_AZIMUTH) and \
                    (np.abs(elevation_bodyframe[i]) <= RADAR_FOR_VERTICAL):
                in_for[i] = 1
                if (range_nm <= self.get_corrected_rdr(np.abs(angle_360_to_180(azimuth_bodyframe[i])))) and \
                        (range_nm > RCPR):
                    in_volume[i] = 1
            zone, tau_mod[i], hmd[i], vmd[i] = hazard_zone_classification(
                ((own_i['X'], own_i['Y'], own_i['Alt(ft)']), (int_i['X'], int_i['Y'], int_i['Alt(ft)'])),
                (own_i['Bearing(Deg)'], int_i['Bearing(Deg)']),
                (own_i['GrndSpd(knots)'], int_i['GrndSpd(knots)']),
                (own_i['DownVel(FPM)'], int_i['DownVel(FPM)']),
                int_i['HorizRangeFt'],
                int_i['HorizRangeRtFtSec'])
            if zone == 'HZ':
                hz[i] = 1
            elif zone == 'MZ':
                mz[i] = 1
            else:
                nhz[i] = 1
            zone_corrective, tm, hm, vm = hazard_zone_classification(
                ((own_i['X'], own_i['Y'], own_i['Alt(ft)']), (int_i['X'], int_i['Y'], int_i['Alt(ft)'])),
                (own_i['Bearing(Deg)'], int_i['Bearing(Deg)']),
                (own_i['GrndSpd(knots)'], int_i['GrndSpd(knots)']),
                (own_i['DownVel(FPM)'], int_i['DownVel(FPM)']),
                int_i['HorizRangeFt'],
                int_i['HorizRangeRtFtSec'],
                'corrective', True)
            zone_waring, tm, hm, vm = hazard_zone_classification(
                ((own_i['X'], own_i['Y'], own_i['Alt(ft)']), (int_i['X'], int_i['Y'], int_i['Alt(ft)'])),
                (own_i['Bearing(Deg)'], int_i['Bearing(Deg)']),
                (own_i['GrndSpd(knots)'], int_i['GrndSpd(knots)']),
                (own_i['DownVel(FPM)'], int_i['DownVel(FPM)']),
                int_i['HorizRangeFt'],
                int_i['HorizRangeRtFtSec'],
                'warning', True)
            if zone_corrective == 'HZ':
                corrective_alert[i] = 1
            if zone_waring == 'HZ':
                warning_alert[i] = 1

        int_df['AzimuthDeg'] = azimuth_bodyframe
        int_df['VertAngleDeg'] = elevation_bodyframe
        int_df['InFOR'] = in_for
        int_df['InRadarVolume'] = in_volume
        int_df['HZ'] = hz
        int_df['MZ'] = mz
        int_df['NHZ'] = nhz
        int_df['TauMod'] = tau_mod
        int_df['HMD'] = hmd
        int_df['VMD'] = vmd
        int_df['CorrectiveAlert'] = corrective_alert
        int_df['WarningAlert'] = warning_alert

    def get_corrected_rdr(self, bearing):
        for b, f in zip(BEARING_CUTOFFS, CORRECTION_FACTORS[self.intruder.size]):
            if np.abs(bearing) < b:
                return self.intruder.base_rdr * f

    def set_interp_df_radar(self):
        self.ownship.truth_interp_radar_df = self.ownship.get_truth_interpolated(
            self.intruder.radar_df, ['EastVel(knots)', 'NorthVel(knots)', 'Pitch(Deg)', 'Roll(Deg)'])
        self.intruder.truth_interp_radar_df = self.intruder.get_truth_interpolated(
            self.intruder.radar_df, True)

    def set_interp_df_adsb(self):
        self.ownship.truth_interp_self_adsb_df = self.ownship.get_truth_interpolated(
            self.ownship.adsb_df, ['LatD', 'LonD', 'Alt(ft)', 'EastVel(knots)', 'NorthVel(knots)',
                                   'DownVel(FPM)', 'Bearing(Deg)', 'Pitch(Deg)', 'Roll(Deg)'])
        self.intruder.truth_interp_self_adsb_df = self.intruder.get_truth_interpolated(
            self.intruder.adsb_df, ['LatD', 'LonD', 'Alt(ft)', 'EastVel(knots)', 'NorthVel(knots)',
                                    'DownVel(FPM)', 'Bearing(Deg)', 'Pitch(Deg)', 'Roll(Deg)'])

    def set_interp_df_tracker(self):
        self.ownship.truth_interp_tracker_df = self.ownship.get_truth_interpolated(self.intruder.tracker_df, True)
        self.intruder.truth_interp_tracker_df = self.intruder.get_truth_interpolated(self.intruder.tracker_df, True)

    def plot_positions(self, label='XXX', plot_path=None):
        import os
        own_df = self.ownship.truth_df
        int_df = self.intruder.truth_df
        plt.figure()
        wcv_times = self.intruder.truth_df.query('HZ == 1').index
        for t in wcv_times[::5]:
            plt.plot(feet_to_nmi(np.array([own_df.loc[t]['X'], int_df.loc[t]['X']])),
                     feet_to_nmi(np.array([own_df.loc[t]['Y'], int_df.loc[t]['Y']])),
                     color='r')
        plt.plot(feet_to_nmi(own_df['X']), feet_to_nmi(own_df['Y']), color='b', linewidth=3)
        plt.plot(feet_to_nmi(int_df['X']), feet_to_nmi(int_df['Y']), color='.15', linewidth=3)
        plt.scatter(feet_to_nmi(own_df.iloc[0]['X']), feet_to_nmi(own_df.iloc[0]['Y']), s=60, color='b')
        plt.scatter(feet_to_nmi(int_df.iloc[0]['X']), feet_to_nmi(int_df.iloc[0]['Y']), s=60, color='.15')
        plt.xlabel('X (nmi)')
        plt.ylabel('Y (nmi)')
        plt.axis('equal')
        if plot_path is not None:
            plt.savefig(os.path.join(plot_path, label), dpi=100)
            plt.close()

    def plot_bearings(self, relative=True):
        if relative:
            bearing_key = 'RelAzimuthDeg'
        else:
            bearing_key = 'AzimuthDeg'
        own_df = self.ownship.truth_interp_radar_df
        int_df = self.intruder.truth_interp_radar_df
        plt.figure()
        plt.plot(own_df['LonD'], own_df['LatD'], color='b')
        plt.plot(int_df['LonD'], int_df['LatD'], color='r')
        plt.scatter(own_df.iloc[0]['LonD'], own_df.iloc[0]['LatD'], color='b')
        plt.scatter(int_df.iloc[0]['LonD'], int_df.iloc[0]['LatD'], color='r')
        step_size = int(len(int_df) / 6)
        for i in range(10, len(int_df), step_size):
            plt.plot([own_df.iloc[i]['LonD'], int_df.iloc[i]['LonD']],
                     [own_df.iloc[i]['LatD'], int_df.iloc[i]['LatD']], color='k')
            plt.text(own_df.iloc[i]['LonD'], own_df.iloc[i]['LatD'], str(np.round(int_df.iloc[i][bearing_key], 1)))
        plt.axis('equal')

    def plot_in_volume(self):
        own_df = self.ownship.truth_df
        int_df = self.intruder.truth_df
        plt.figure()
        plt.plot(own_df['LonD'], own_df['LatD'], color='b')
        int_df_in_volume = int_df[int_df['InRadarVolume'] == 1]
        int_df_out_volume = int_df[int_df['InRadarVolume'] == 0]
        plt.scatter(int_df_out_volume['LonD'], int_df_out_volume['LatD'], color='r')
        plt.scatter(int_df_in_volume['LonD'], int_df_in_volume['LatD'], color='g')
        plt.scatter(own_df.iloc[0]['LonD'], own_df.iloc[0]['LatD'], color='b')
        plt.scatter(int_df.iloc[0]['LonD'], int_df.iloc[0]['LatD'], color='k')
        plt.axis('equal')

    class NoAdsbDataException(IndexError):
        def __init__(self):
            IndexError.__init__(self, 'There is no ADS-B data for this intruder.')

    def adsb_report(self, label='XXX', log_path=None):
        import os
        import pandas as pd
        from position_calcs import angle_difference
        from conversions import angle_nav_to_math, angle_360_to_180

        truth_own_df = self.ownship.truth_interp_self_adsb_df
        truth_int_df = self.intruder.truth_interp_self_adsb_df
        adsb_own_df = self.ownship.adsb_df
        adsb_int_df = self.intruder.adsb_df

        if (type(adsb_own_df) == type(None)) or (adsb_own_df.shape[0] == 0):
            raise self.NoAdsbDataException
        own_errors = {}
        own_errors['Latitude(deg)'] = adsb_own_df['Latitude(Deg)'].values - truth_own_df['LatD'].values
        own_errors['Longitude(deg)'] = adsb_own_df['Longitude(Deg)'].values - truth_own_df['LonD'].values
        own_errors['Altitude(ft)'] = (adsb_own_df['Altitude(100ft)'].values * 100) - truth_own_df['Alt(ft)'].values
        own_errors['EastVel(kts)'] = (adsb_own_df['Speed(kts)'].values * np.cos(
            np.deg2rad(angle_nav_to_math(adsb_own_df['Heading(Deg)'].values)))) - \
            truth_own_df['EastVel(knots)'].values
        own_errors['NorthVel(kts)'] = (adsb_own_df['Speed(kts)'].values * np.sin(
            np.deg2rad(angle_nav_to_math(adsb_own_df['Heading(Deg)'].values)))) - \
            truth_own_df['NorthVel(knots)'].values
        own_errors['UpVel(fpm)'] = adsb_own_df['GeoRate(ft/min)'].values + truth_own_df['DownVel(FPM)'].values
        own_errors['Yaw(deg)'] = angle_difference(adsb_own_df['Yaw(Deg)'].values,
                                                  angle_360_to_180(truth_own_df['Bearing(Deg)'].values))
        own_errors['Pitch(deg)'] = angle_difference(adsb_own_df['Pitch(deg)'].values, truth_own_df['Pitch(Deg)'].values)
        own_errors['Roll(deg)'] = angle_difference(adsb_own_df['Roll(Deg)'].values, truth_own_df['Roll(Deg)'].values)
        own_error_df = pd.DataFrame.from_dict(own_errors)
        own_error_df['M3A'] = 1
        own_error_df['TimeSecs'] = adsb_own_df.index.values
        int_errors = {}
        int_errors['Latitude(deg)'] = adsb_int_df['Latitude(Deg)'].values - truth_int_df['LatD'].values
        int_errors['Longitude(deg)'] = adsb_int_df['Longitude(Deg)'].values - truth_int_df['LonD'].values
        int_errors['Altitude(ft)'] = (adsb_int_df['Altitude(100ft)'].values * 100) - truth_int_df['Alt(ft)'].values
        int_errors['EastVel(kts)'] = (adsb_int_df['Speed(kts)'].values * np.cos(
            np.deg2rad(angle_nav_to_math(adsb_int_df['Heading(Deg)'].values)))) - \
            truth_int_df['EastVel(knots)'].values
        int_errors['NorthVel(kts)'] = (adsb_int_df['Speed(kts)'].values * np.sin(
            np.deg2rad(angle_nav_to_math(adsb_int_df['Heading(Deg)'].values)))) - \
            truth_int_df['NorthVel(knots)'].values
        int_errors['UpVel(fpm)'] = adsb_int_df['GeoRate(ft/min)'].values + truth_int_df['DownVel(FPM)'].values
        int_error_df = pd.DataFrame.from_dict(int_errors)
        int_error_df['M3A'] = 2
        int_error_df['TimeSecs'] = adsb_int_df.index.values
        error_df = pd.concat((own_error_df, int_error_df))
        error_df.set_index(['M3A', 'TimeSecs'], inplace=True)
        error_df.to_csv(os.path.join(log_path, label + '.csv'))

    class NoRadarDataException(IndexError):
        def __init__(self):
            IndexError.__init__(self, 'There is no radar data for this intruder.')

    def radar_report(self, label='XXX', log_path=None, plot_path_values=None, plot_path_errors=None):
        """Produces radar error report plots and logs for the intruder aircraft.
           aircraft: Aircraft object with a radar DataFrame and truth DF interpolated with entries
                     corresponding to radar hit times
           label: string label for figure
           log_path: directory to save error log DataFrame to
           plot_path_values: directory to save values plot to
           plot_path_errors: directory to save errors plot to"""

        import os
        import pandas as pd
        from conversions import angle_360_to_180
        from position_calcs import angle_difference, bearing_from_en_coords
        from coord_transforms import rotate_3d

        truth_df = self.intruder.truth_interp_radar_df
        radar_df = self.intruder.radar_df
        own_truth_df = self.ownship.truth_interp_radar_df

        if (type(radar_df) == type(None)) or (radar_df.shape[0] == 0):
            raise self.NoRadarDataException

        own_truth_df['Bearing(Deg)'] = bearing_from_en_coords(
            (own_truth_df['EastVel(knots)'].values, own_truth_df['NorthVel(knots)'].values))

        for i in range(truth_df.shape[0]):
            own_yaw, own_pitch, own_roll = (angle_360_to_180(own_truth_df.iloc[i]['Bearing(Deg)']),
                                            own_truth_df.iloc[i]['Pitch(Deg)'], own_truth_df.iloc[i]['Roll(Deg)'])
            enu_body_frame = rotate_3d((truth_df.iloc[i]['EFt'], truth_df.iloc[i]['NFt'], truth_df.iloc[i]['UFt']),
                                       own_yaw, own_pitch, own_roll)
            truth_df['AzimuthDeg'].iloc[i] = bearing_from_en_coords(enu_body_frame[:2])

        errors = {}
        errors['Range'] = nmi_to_feet(radar_df['Range(NMI)'].values) - truth_df['RangeFt'].values
        errors['Azimuth'] = angle_difference(radar_df['Azimuth(Deg)'].values, truth_df['AzimuthDeg'].values)
        errors['AzimuthUncorrected'] = angle_difference(radar_df['Azimuth(Deg)'].values,
                                                        truth_df['AzimuthBFUncorrectedDeg'].values)
        errors['Elevation'] = radar_df['ElevAngle(Deg)'].values - truth_df['VertAngleDeg'].values
        error_units = {'Range': 'ft', 'Azimuth': 'deg', 'Elevation': 'deg'}

        truth_key_map = {'Range': 'RangeFt',
                         'Azimuth': 'AzimuthDeg', 'Elevation': 'VertAngleDeg'}

        def get_truth_vals(param):
            if (param == 'Range') or (param == 'Range Rate'):
                return feet_to_nmi(self.intruder.truth_df[truth_key_map[param]].values)
            else:
                return self.intruder.truth_df[truth_key_map[param]].values

        radar_key_map = {'Range': 'Range(NMI)', 'Azimuth': 'Azimuth(Deg)', 'Elevation': 'ElevAngle(Deg)'}

        def get_radar_vals(param):
            return radar_df[radar_key_map[param]].values

        value_units = {'Range': 'nmi', 'Azimuth': 'deg', 'Elevation': 'deg'}

        def plot_value_by_time(param, subplot_index):
            """param: str of parameter ("Range", "Horizontal Angle", etc.)
               subplot_index: str of matplotlib figure subplot"""
            plt.subplot(subplot_index)

            plt.plot(self.intruder.truth_df.index, get_truth_vals(param), alpha=.8, zorder=1)
            plt.scatter(radar_df.index, get_radar_vals(param), s=8, color='r', alpha=.5, zorder=2)
            plt.xlabel('Time (s)', fontsize=8)
            plt.xlim((0, self.intruder.truth_df.index.max()))
            plt.ylabel(param + ' (' + value_units[param] + ')', fontsize=8)
            plt.title(param + ' Over Time', y=1.08, fontsize=9)

        def plot_error_by_time(param, subplot_index):
            plt.subplot(subplot_index)
            plt.plot(x, errors[param])
            plt.tick_params(axis='both', which='major', labelsize=7)
            plt.xlim(x_lims)
            plt.xlabel('Time (s)', fontsize=8)
            plt.ylabel('Error (' + error_units[param] + ')', fontsize=8)
            plt.title(param + ' Error Over Time', y=1.08, fontsize=9)

        def plot_error_hist(param, subplot_index):
            ax = plt.subplot(subplot_index)
            vals = errors[param]
            bins = np.linspace(vals.min(), vals.max(), len(vals) // 2 + 2)
            plt.hist(errors[param], bins=bins, linewidth=1, orientation=u'horizontal', edgecolor='none')
            plt.tick_params(axis='both', which='major', labelsize=7)
            plt.xlabel('# of Sensor Readings', fontsize=8)
            plt.ylabel('Error (' + error_units[param] + ')', fontsize=8)
            mean_error = np.around(np.mean(vals), 2)
            sd_error = np.around(np.std(vals, ddof=0), 2)
            summary_text_string = 'Mean = ' + str(mean_error) + '\n' + 'SD = ' + str(sd_error)
            sum_text = ax.text(.9, .9, summary_text_string, transform=ax.transAxes,
                               ha='right', va='top', fontsize=8)
            sum_text.set_bbox(dict(color='.75', alpha=0.8, edgecolor='.75'))
            plt.title('Distribution of ' + param + ' Errors', y=1.08, fontsize=9)

        if log_path is not None:
            error_df = pd.DataFrame.from_dict(errors)
            error_df.set_index(radar_df.index, inplace=True)
            error_df.to_csv(os.path.join(log_path, label + '.csv'))

        if plot_path_values is not None:
            fig = plt.figure()
            plot_value_by_time('Range', '321')
            plot_value_by_time('Azimuth', '323')
            plot_value_by_time('Elevation', '324')
            fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.85)
            plt.suptitle(label, fontsize=12)
            plt.savefig(os.path.join(plot_path_values, label), dpi=300)
            plt.close()

        if plot_path_errors is not None:
            fig = plt.figure()
            x = radar_df.index
            x_range = x[-1] - x[0]
            x_lims = (x[0] - .05 * x_range, x[-1] + .05 * x_range)

            plot_error_by_time('Range', '421')
            plot_error_hist('Range', '422')
            plot_error_by_time('Azimuth', '425')
            plot_error_hist('Azimuth', '426')
            plot_error_by_time('Elevation', '427')
            plot_error_hist('Elevation', '428')

            fig.subplots_adjust(hspace=1.0, wspace=0.3, top=0.9)
            plt.suptitle(label, fontsize=12)
            plt.savefig(os.path.join(plot_path_errors, label), dpi=400)
            plt.close()

        return errors

    class NoTrackerDataException(IndexError):
        def __init__(self):
            IndexError.__init__(self, 'There is no tracker data for this intruder.')

    def tracker_report(self, label='XXX', log_path_alerts=None, log_path_errors=None, plot_path_values=None,
                       plot_path_errors=None):
        """
        Produces tracker report plots and logs for the intruder aircraft.
        :param label: Plot label
        :param log_path_alerts: Directory to save alert log to
        :param log_path_errors: Directory to save error log to
        :param plot_path_values: Directory to save values plot to
        :param plot_path_errors: Directory to save error plot to
        :return: None
        """

        import os
        import pandas as pd
        from conversions import geod_2_enu_pos, meters_to_feet, mps_to_knots, nmips_to_knots
        from position_calcs import angle_difference, bearing_from_en_coords, hazard_zone_classification

        # get tracker data frame and corresponding intruder truth and ownship tracker DFs
        truth_df = self.intruder.truth_interp_tracker_df
        tracker_df = self.intruder.tracker_df
        if (type(tracker_df) == type(None)) or (tracker_df.shape[0] == 0):
            print('no intruder tracker data')
            raise self.NoTrackerDataException
        own_truth_df = self.ownship.truth_interp_tracker_df
        own_tracker_df = self.ownship.tracker_df.loc[tracker_df.index]

        # calculate additional parameters from tracker values
        tracker_enu = geod_2_enu_pos(
            (tracker_df['Lat(Deg)'].values, tracker_df['Lon(Deg)'].values, tracker_df['BaroAlt(ft)'].values),
            (own_tracker_df['Lat(Deg)'].values, own_tracker_df['Lon(Deg)'].values,
             own_tracker_df['BaroAlt(ft)'].values),
            'ft')
        tracker_df['EastFt'] = tracker_enu[0]
        tracker_df['NorthFt'] = tracker_enu[1]
        tracker_df['HorizRangeFt'] = np.sqrt(tracker_enu[0] ** 2 + tracker_enu[1] ** 2)
        tracker_df['AzimuthDeg'] = bearing_from_en_coords(tracker_enu[:2])
        tracker_df['VerticalAngleDeg'] = np.rad2deg(
            np.arctan2(tracker_enu[2], np.sqrt(tracker_enu[0] ** 2 + tracker_enu[1] ** 2)))
        tracker_df['RelAltFt'] = tracker_df['BaroAlt(ft)'].values - own_tracker_df['BaroAlt(ft)'].values
        tracker_df['HeadingDeg'] = bearing_from_en_coords(
            (tracker_df['EDot(nmi/s)'].values, tracker_df['NDot(nmi/s)'].values))
        tracker_df['GroundSpeedKts'] = nmips_to_knots(
            np.sqrt(tracker_df['EDot(nmi/s)'].values ** 2 + tracker_df['NDot(nmi/s)'].values ** 2))

        if log_path_alerts is not None:
            own_heading = bearing_from_en_coords(
                (own_tracker_df['EDot(nmi/s)'].values, own_tracker_df['NDot(nmi/s)'].values))
            own_g_speed_kts = mps_to_knots(
                np.sqrt(own_tracker_df['EDot(nmi/s)'].values ** 2 + own_tracker_df['NDot(nmi/s)'].values ** 2))
            horiz_range_rate = (
                tracker_enu[1] * meters_to_feet(
                    tracker_df['NDot(nmi/s)'].values - own_tracker_df['NDot(nmi/s)'].values) +
                tracker_enu[0] * meters_to_feet(
                    tracker_df['EDot(nmi/s)'].values - own_tracker_df['EDot(nmi/s)'].values) /
                tracker_df['HorizRangeFt'].values)

            # calculate alerts
            alert_log = pd.DataFrame(
                {'CorrectiveAlert': np.zeros(tracker_df.shape[0]), 'WarningAlert': np.zeros(tracker_df.shape[0])},
                index=tracker_df.index)
            for i in range(tracker_df.shape[0]):
                zone_corrective, tau_mod, hmd, vmd = hazard_zone_classification(
                    ((0, 0, own_tracker_df.iloc[i]['BaroAlt(ft)']),
                     (tracker_enu[0][i], tracker_enu[1][i], tracker_df.iloc[i]['BaroAlt(ft)'])),
                    (own_heading[i], tracker_df.iloc[i]['HeadingDeg']),
                    (own_g_speed_kts[i], tracker_df.iloc[i]['GroundSpeedKts']),
                    (own_tracker_df.iloc[i]['AltRate(ft/min)'] / 60, tracker_df.iloc[i]['AltRate(ft/min)'] / 60),
                    tracker_df.iloc[i]['HorizRangeFt'],
                    horiz_range_rate[i], 'corrective', True)
                zone_warning, tau_mod, hmd, vmd = hazard_zone_classification(
                    ((0, 0, own_tracker_df.iloc[i]['BaroAlt(ft)']),
                     (tracker_enu[0][i], tracker_enu[1][i], tracker_df.iloc[i]['BaroAlt(ft)'])),
                    (own_heading[i], tracker_df.iloc[i]['HeadingDeg']),
                    (own_g_speed_kts[i], tracker_df.iloc[i]['GroundSpeedKts']),
                    (own_tracker_df.iloc[i]['AltRate(ft/min)'] / 60, tracker_df.iloc[i]['AltRate(ft/min)'] / 60),
                    tracker_df.iloc[i]['HorizRangeFt'],
                    horiz_range_rate[i], 'warning', True)
                if zone_corrective == 'HZ':
                    alert_log.iloc[i]['CorrectiveAlert'] = 1
                if zone_warning == 'HZ':
                    alert_log.iloc[i]['WarningAlert'] = 1
            alert_log.to_csv(os.path.join(log_path_alerts, label + 'tracker_alerts.csv'))

        errors = {}
        errors['TrackID'] = tracker_df['TrkNum'].values
        errors['East(ft)'] = tracker_df['EastFt'].values - truth_df['EFt'].values
        errors['North(ft)'] = tracker_df['NorthFt'].values - truth_df['NFt'].values
        errors['Horizontal Position(ft)'] = np.sqrt(errors['East(ft)'] ** 2 + errors['North(ft)'] ** 2)
        errors['Horizontal Range(ft)'] = np.sqrt(tracker_enu[0] ** 2 + tracker_enu[1] ** 2) - np.sqrt(
            truth_df['EFt'].values ** 2 + truth_df['NFt'].values ** 2)
        errors['Altitude(ft)'] = tracker_df['BaroAlt(ft)'].values - truth_df['Alt(ft)'].values
        errors['Relative Altitude(ft)'] = (tracker_df['BaroAlt(ft)'].values - own_tracker_df['BaroAlt(ft)'].values) - \
            truth_df['RelAltFt'].values
        errors['Heading(deg)'] = angle_difference(tracker_df['HeadingDeg'].values, truth_df['Bearing(Deg)'].values)
        errors['Ground Speed(kts)'] = tracker_df['GroundSpeedKts'].values - truth_df['GrndSpd(knots)'].values
        errors['Vertical Rate(ft/min)'] = tracker_df['AltRate(ft/min)'].values + truth_df['DownVel(FPM)'].values
        errors['RelHorizVelocity(kts)'] = np.sqrt(
            (nmips_to_knots(tracker_df['NDot(nmi/s)'].values - own_tracker_df['NDot(nmi/s)'].values) -
             (truth_df['NorthVel(knots)'].values - own_truth_df['NorthVel(knots)'].values)) ** 2 +
            (nmips_to_knots(tracker_df['EDot(nmi/s)'].values - own_tracker_df['EDot(nmi/s)'].values) -
             (truth_df['EastVel(knots)'].values - own_truth_df['EastVel(knots)'].values)) ** 2)
        for v_flag in ('HPosValid', 'VPosValid', 'HVelValid', 'VVelValid'):
            errors[v_flag] = tracker_df[v_flag].values
        errors['Valid'] = tracker_df['HPosValid'].values * tracker_df['VPosValid'].values * \
            tracker_df['HVelValid'].values * tracker_df['VVelValid'].values

        error_key_map = {'East': 'East(ft)', 'North': 'North(ft)', 'Horizontal Range': 'Horizontal Range(ft)',
                         'Horizontal Position': 'Horizontal Position(ft)', 'Relative Altitude': 'Relative Altitude(ft)',
                         'Heading': 'Heading(deg)', 'Ground Speed': 'Ground Speed(kts)',
                         'Vertical Rate': 'Vertical Rate(ft/min)', 'Horizontal Velocity': 'RelHorizVelocity(kts)'}
        error_units = {'East': 'ft', 'North': 'ft', 'Horizontal Range': 'ft', 'Horizontal Position': 'ft',
                       'Relative Altitude': 'ft', 'Azimuth': 'deg', 'Heading': 'deg', 'Ground Speed': 'kts',
                       'Vertical Rate': 'ft/min', 'Horizontal Velocity': 'kts'}
        truth_key_map = {'East': 'EFt', 'North': 'NFt', 'Horizontal Range': 'HorizRangeFt',
                         'Altitude': 'Alt(ft)', 'Relative Altitude': 'RelAltFt',
                         'Heading': 'Bearing(Deg)', 'Ground Speed': 'GrndSpd(knots)', 'Vertical Rate': 'DownVel(FPM)'}

        def get_truth_vals(param):
            if param == 'Vertical Rate':
                return -1 * self.intruder.truth_df[truth_key_map[param]].values
            else:
                return self.intruder.truth_df[truth_key_map[param]].values

        tracker_key_map = {'East': 'EastFt', 'North': 'NorthFt', 'Horizontal Range': 'HorizRangeFt',
                           'Altitude': 'BaroAlt(ft)', 'Relative Altitude': 'RelAltFt',
                           'Ground Speed': 'GroundSpeedKts', 'Vertical Rate': 'AltRate(ft/min)',
                           'Heading': 'HeadingDeg'}

        def get_tracker_vals(param):
            return tracker_df[tracker_key_map[param]].values

        value_units = error_units

        def plot_value_by_time(param, subplot_index):
            """
            :param param: str of parameter ("Range", "Horizontal Angle", etc.)
            :param subplot_index: str of matplotlib figure subplot
            :return: None
            """
            plt.subplot(subplot_index)

            plt.plot(self.intruder.truth_df.index, get_truth_vals(param), alpha=.8, zorder=1)
            plt.scatter(tracker_df.index, get_tracker_vals(param), s=8, color='r', alpha=.5, zorder=2)
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.xlabel('Time (s)', fontsize=8)
            plt.xlim((0, self.intruder.truth_df.index.max()))
            plt.ylabel(param + ' (' + value_units[param] + ')', fontsize=8)
            plt.title(param + ' Over Time', y=1.08, fontsize=9)

        def plot_error_by_time(param, fig, subplot_index):
            row, col, index = subplot_index
            fig.add_subplot(row, col, index)
            plt.plot(x, errors[error_key_map[param]])
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.xlim(x_lims)
            plt.xlabel('Time (s)', fontsize=8)
            plt.ylabel('Error (' + error_units[param] + ')', fontsize=8)
            plt.title(param + ' Error Over Time', y=1.08, fontsize=9)

        def plot_error_hist(param, fig, subplot_index):
            row, col, index = subplot_index
            ax = fig.add_subplot(row, col, index)
            vals = errors[error_key_map[param]]
            bins = np.linspace(vals.min(), vals.max(), len(vals) // 2 + 2)
            plt.hist(vals, bins=bins, linewidth=1, orientation=u'horizontal', edgecolor='none')
            plt.tick_params(axis='both', which='major', labelsize=6)
            plt.xlabel('# of Sensor Readings', fontsize=8)
            plt.ylabel('Error (' + error_units[param] + ')', fontsize=8)
            mean_error = np.around(np.mean(vals), 2)
            sd_error = np.around(np.std(vals, ddof=0), 2)
            summary_text_string = 'Mean = ' + str(mean_error) + '\n' + 'SD = ' + str(sd_error)
            sum_text = ax.text(.9, .9, summary_text_string, transform=ax.transAxes,
                               ha='right', va='top', fontsize=8)
            sum_text.set_bbox(dict(color='.75', alpha=0.8, edgecolor='.75'))
            plt.title('Distribution of ' + param + ' Errors', y=1.08, fontsize=9)

        if log_path_errors is not None:
            error_df = pd.DataFrame.from_dict(errors)
            error_df.set_index(tracker_df.index, inplace=True)
            error_df.to_csv(os.path.join(log_path_errors, label + '.csv'))

        if plot_path_values is not None:
            fig = plt.figure(figsize=(8, 9))
            plot_value_by_time('East', '321')
            plot_value_by_time('North', '322')
            plot_value_by_time('Relative Altitude', '323')
            plot_value_by_time('Vertical Rate', '324')
            plot_value_by_time('Ground Speed', '325')
            plot_value_by_time('Heading', '326')
            fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.85)
            plt.suptitle(label, fontsize=12)
            plt.savefig(os.path.join(plot_path_values, label), dpi=300)
            plt.close()

        if plot_path_errors is not None:
            fig = plt.figure(figsize=(8, 10))
            x = tracker_df.index
            x_range = x[-1] - x[0]
            x_lims = (x[0] - .05 * x_range, x[-1] + .05 * x_range)

            plot_error_by_time('Horizontal Position', fig, (6, 2, 1))
            plot_error_hist('Horizontal Position', fig, (6, 2, 2))
            plot_error_by_time('Relative Altitude', fig, (6, 2, 3))
            plot_error_hist('Relative Altitude', fig, (6, 2, 4))
            plot_error_by_time('Horizontal Velocity', fig, (6, 2, 5))
            plot_error_hist('Horizontal Velocity', fig, (6, 2, 6))
            plot_error_by_time('Vertical Rate', fig, (6, 2, 7))
            plot_error_hist('Vertical Rate', fig, (6, 2, 8))
            plot_error_by_time('Ground Speed', fig, (6, 2, 9))
            plot_error_hist('Ground Speed', fig, (6, 2, 10))
            plot_error_by_time('Heading', fig, (6, 2, 11))
            plot_error_hist('Heading', fig, (6, 2, 12))

            fig.subplots_adjust(hspace=1.0, wspace=0.3, top=0.9)
            plt.suptitle(label, fontsize=12)
            plt.savefig(os.path.join(plot_path_errors, label), dpi=400)
            plt.close()

        return errors
