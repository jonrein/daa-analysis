import numpy as np
from conversions import angle_nav_to_math, knots_to_fps, nmi_to_feet


def bearing_from_en_coords(en):
    """
    Given a tuple of east-north relative coordinates (values can be scalars
       or ndarrays), returns the bearing of the coordinate(s) and the
       (ownship) origin in degrees, 0-360
    :param en: tuple of east-north relative coordinates
    :return: bearing(s)
    """
    bearing = np.rad2deg(np.arctan2(en[0], en[1]))
    if hasattr(bearing, '__iter__'):
        negs = bearing < 0
        bearing[negs] += 360
        return bearing
    else:
        if bearing < 0:
            return 360 + bearing
        else:
            return bearing


def relative_bearing_from_en_coords(en, own_heading):
    """
    Given a tuple of east-north relative coordinates (values can be scalars
       or ndarrays), returns the bearing of the coordinate(s) and the
       (ownship) origin in degrees, relative to the provided ownship heading
    :param en: east-north coordinates
    :param own_heading: ownship heading (in degrees)
    :return: relative bearing (in degrees)
    """
    from conversions import angle_360_to_180
    return angle_360_to_180(bearing_from_en_coords(en) - own_heading)


def angle_difference(theta, ref_theta):
    return (theta + 180 - ref_theta) % 360 - 180


def modified_tau(horiz_range, horiz_range_rate, dmod):
    r = horiz_range
    dr = horiz_range_rate
    if r <= dmod:
        return 0
    elif (dr >= 0) and (r > dmod):
        return np.inf
    else:
        return (dmod ** 2 - r ** 2) / (r * dr)


def miss_distance(positions, headings, ground_speeds, vertical_rates):
    """
    Returns the projected dead-reckoning miss distance between two aircraft
       given their current positions, headings, ground speeds, and vertical rates.
    :param positions: tuple of x,y,z (e.g. ENU) coordinate tuples, in feet, one per aircraft
    :param headings: tuple of headings in degrees, one value per aircraft
    :param ground_speeds: tuple of ground speeds in knots, one value per aircraft
    :param vertical_rates: tuple of vertical rates in feet/min, one value per aircraft
    :return: tuple of (horizontal miss distance in feet, vertical miss distance in feet, time of slant range
     closest point of approach in seconds)
    """

    pos1_start, pos2_start = positions
    head1, head2 = (np.deg2rad(angle_nav_to_math(headings[0])),
                    np.deg2rad(angle_nav_to_math(headings[1])))
    g_speed1, g_speed2 = (knots_to_fps(ground_speeds[0]), knots_to_fps(ground_speeds[1]))
    v_rate1, v_rate2 = (vertical_rates[0] / 60, vertical_rates[1] / 60)  # converting to feet/sec
    
    pos_diff_start = np.array(pos1_start) - np.array(pos2_start)
    vector1 = np.vstack((g_speed1 * np.array([[np.cos(head1)], [np.sin(head1)]]),
                         v_rate1))
    vector2 = np.vstack((g_speed2 * np.array([[np.cos(head2)], [np.sin(head2)]]),
                         v_rate2))
    vector_diff = vector2 - vector1

    cpa_time = np.dot(pos_diff_start.T, vector_diff)[0]/np.dot(vector_diff.T, vector_diff)[0][0]
    diff = (pos1_start + vector1.T*cpa_time) - (pos2_start + vector2.T*cpa_time)
    cpa_horiz_range = np.linalg.norm(diff[0][:2])
    cpa_alt = diff[0][2]
    return cpa_horiz_range, cpa_alt, cpa_time


def vertical_closure_rate(own_info, int_info):
    """own_info: tuple of ownship's altitude and altitude rate
       int_info: tuple of intruder's altitude and altitude rate
       returns: vertical closure rate between the aircraft in units of provided
                altitude rate (positive values indicate vertically converging)"""
    from numpy import sign
    own_alt, own_alt_rate = own_info
    int_alt, int_alt_rate = int_info
    return (own_alt_rate-int_alt_rate) * sign(int_alt-own_alt)

# Hazard zone and alert classification ##############################
tau_mod_crit = {'HAZ': {'preventive': 35, 'corrective': 35, 'warning': 35},
                'HAZNot': {'preventive': 110, 'corrective': 110, 'warning': 90}}
dmod = {'HAZ': {'preventive': nmi_to_feet(.66), 'corrective': nmi_to_feet(.66), 'warning': nmi_to_feet(.66)},
        'HAZNot': {'preventive': nmi_to_feet(1.5), 'corrective': nmi_to_feet(1.5),
                   'warning': nmi_to_feet(1.2)}}
v_crit = {'HAZ': {'preventive': 700, 'corrective': 450, 'warning': 450},
          'HAZNot': {'preventive': 800, 'corrective': 450, 'warning': 450}}
algo_tau_mod_deltas = {'corrective': 75, 'warning': 65}
algo_dmod_deltas = {'corrective': 2000, 'warning': 1000}  # ft


def hazard_zone_classification(positions, headings, ground_speeds, vertical_rates,
                               horiz_range, horiz_range_rate, alert_type='corrective',
                               algorithm=False):
    """
    Returns a hazard zone classification, modified tau for the hazard zone, and projected
       miss distances for two aircraft, given their current positions, headings, groundSpeeds,
       vertical rates, horizontal range and horizontal range rate.  For all tuple parameters,
       ownship is given first.
    :param positions: tuple of x,y,z UTM coordinate (e.g. ENU) tuples, in feet, one per aircraft
    :param headings: tuple of headings in degrees, one value per aircraft
    :param ground_speeds: tuple of ground speeds in knots, one value per aircraft
    :param vertical_rates: tuple of vertical rates in feet/sec, one value per aircraft
    :param horiz_range: horizontal range between aircraft in feet
    :param horiz_range_rate: horizontal range rate for the two aircraft in feet/sec
    :param alert_type: type of alert for zone classification (preventive, corrective, or warning)
    :param algorithm: Boolean indicating whether adding a delta on tau_mod_crit for algorithm RA
    :return: tuple of zone classification (string) modified tau (float), horizontal miss disance (float),
        and vertical miss distance (float)
    """
    hmd, vmd, cpa_time = miss_distance(positions, headings, ground_speeds, vertical_rates)
    if cpa_time < 0:
        hmd = horiz_range
    
    v_dist = np.abs(positions[0][-1] - positions[1][-1])
    v_rate1, v_rate2 = vertical_rates
    v_closure_rate = vertical_closure_rate((positions[0][-1], v_rate1),
                                           (positions[1][-1], v_rate2))

    if algorithm:
        alg_tau_mod_crit = tau_mod_crit['HAZ'][alert_type] + algo_tau_mod_deltas[alert_type]
        alg_dmod = dmod['HAZ'][alert_type] + algo_dmod_deltas[alert_type]
        tau_mod_haz = modified_tau(horiz_range, horiz_range_rate, alg_dmod)
        vmod = v_crit['HAZ'][alert_type]
        v_haz_size = max(vmod, vmod - v_closure_rate*alg_tau_mod_crit)
        if (tau_mod_haz <= alg_tau_mod_crit) and (hmd <= alg_dmod) and (v_dist <= v_haz_size):
            zone = 'HZ'
        else:
            zone = 'NHZ'
    else:
        tau_mod_haz = modified_tau(horiz_range, horiz_range_rate, dmod['HAZ'][alert_type])
        tau_mod_haz_not = modified_tau(horiz_range, horiz_range_rate, dmod['HAZNot'][alert_type])
        vmod = v_crit['HAZNot'][alert_type]
        v_haz_not_size = max(vmod, vmod - v_closure_rate * tau_mod_crit['HAZNot'][alert_type])
        if (tau_mod_haz <= (tau_mod_crit['HAZ'][alert_type])) and (hmd <= dmod['HAZ'][alert_type]) \
           and (v_dist <= v_crit['HAZ'][alert_type]):
            zone = 'HZ'
        elif (tau_mod_haz_not > tau_mod_crit['HAZNot'][alert_type]) or (hmd > dmod['HAZNot'][alert_type]) \
                or (v_dist > v_haz_not_size):
            zone = 'NHZ'
        else:
            zone = 'MZ'
    
    return zone, tau_mod_haz, hmd, vmd


def haz_time_from_truth_df(truth_df):
    try:
        ix = np.nonzero(truth_df['HZ'].values == 1)[0][0]
        time = truth_df.index[ix]
        return time, ix
    except IndexError:
        return None, None


def haznot_leave_time_from_truth_df(truth_df):
    _, haz_ix = haz_time_from_truth_df(truth_df)
    if haz_ix is None:
        end_ix = truth_df.shape[0]
    else:
        end_ix = haz_ix
    try:
        hzn_leave_ix = np.nonzero(truth_df['MZ'].values[:end_ix] == 1)[0][0]
        return truth_df.index[hzn_leave_ix]
    except IndexError:
        try:
            hzn_leave_ix = np.nonzero(truth_df['NHZ'].values[:end_ix] == 1)[0][-1]
            return truth_df.index[hzn_leave_ix]
        except IndexError:
            return None