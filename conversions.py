import numpy as np
from mymods.conversion import lat_long_decimal, time_string_to_num

ECC = 0.081819190842622
EQ_RADIUS_M = 6378137.0
PO_RADIUS_M = 6356752.3142

def meters_to_feet(x):
    return 3.280839895013123 * x

def meters_to_nmi(x):
    return x / 1852

def feet_to_meters(x):
    return x / 3.280839895013123

def feet_to_nmi(x):
    return x / 6076.1155

def nmi_to_feet(x):
    return 6076.1155 * x

def knots_to_fps(x):
    return x * 1.68780986111

def knots_to_mps(x):
    return x * 0.5144444456663281

def mps_to_knots(x):
    return x / 0.5144444456663281

def fps_to_knots(x):
    return x / 1.68780986111

def nmips_to_knots(x):
    return x * 3600

def fpm_to_mps(x):
    return feet_to_meters(x / 60)

def angle_nav_to_math(theta):
    return np.rad2deg(np.arctan2(np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))))

def angle_360_to_180(theta):
    '''takes angle(s) in degrees and puts it in range -180 to 180'''
    return np.rad2deg(np.arctan2(np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))))

def angle_180_to_360(theta):
    '''takes angle(s) in degrees and puts it in range 0 to 360'''
    if hasattr(theta,'__iter__'):
        negs = theta < 0
        theta[negs] = 360 + theta[negs]
        return bearing
    else:
        if theta < 0:
            return 360 + theta
        else:
            return theta

def latlon_to_UTM(lats,lons,utmZone=None,units='ft'):
    from pyproj import Proj
    if utmZone == None:
        utmZone = get_UTM_zone((lats[0],lons[0]))
    elif type(utmZone) == tuple:
        utmZone = get_UTM_zone(utmZone)
    projection = Proj(proj='utm',zone=utmZone,ellps='WGS84')
    xyMeters = list(projection(lons,lats))
    if 'units' == 'm':
        return xyMeters
    else:
        xyFeet = [meters_to_feet(cSet) for cSet in xyMeters]
        return xyFeet[::-1]

def get_UTM_zone(latlon,checkSN=False):
    '''Takes tuple of (lat,lon) and returns UTM zone for conversion.
       If checkSN is True, checks for special cases of Svalbard and Norway.'''
    lat,lon = latlon
    zoneNum = int((lon + 180)/6) + 1
    if checkSN == True:
        if (lat >= 56) and (lat < 64) and (lon >= 3) and (lon < 12):
            zoneNum = 32
        if (lat >= 72) and (lat < 84):
            if (lon >= 0) and (lon < 9):
                zoneNum = 31
            if (lon >= 9) and (lon < 21):
                zoneNum = 33
            if (lon >= 21) and (lon < 33):
                zoneNum = 35
            if (lon >= 33) and (lon < 42):
                zoneNum = 37
    return zoneNum

def geod_2_ecef_pos(geodCoord, units):
    '''assumes "geodCoord" is a (lat,lon,alt) tuple in units of
       deg, deg, and ft, returns x,y,z earth-centered earth-fixed coordinates'''
    alt = geodCoord[2]
    if units == 'm':
        eqRadius = EQ_RADIUS_M
        alt = feet_to_meters(alt)
    elif units == 'ft':
        eqRadius = meters_to_feet(EQ_RADIUS_M)
    elif units == 'nmi':
        eqRadius = meters_to_nmi(EQ_RADIUS_M)
        alt = feet_to_nmi(alt)
    lat = np.deg2rad(geodCoord[0])
    lon = np.deg2rad(geodCoord[1])
    N = eqRadius / np.sqrt(1 - ECC**2 * np.sin(lat)**2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1-ECC**2) + alt) * np.sin(lat)
    return (x, y, z)

def geod_2_enu_pos(targetGeod, refGeod, units='ft'):
    '''assumes "targetGeod" and "refGeod" are (lat,lon,alt) tuples in units of
       deg, deg, and ft, returns east, north, up coordinates'''
    targetECEF = geod_2_ecef_pos(targetGeod, units)
    refECEF = geod_2_ecef_pos(refGeod, units)
    dx = targetECEF[0] - refECEF[0]
    dy = targetECEF[1] - refECEF[1]
    dz = targetECEF[2] - refECEF[2]
    sinLat = np.sin(np.deg2rad(refGeod[0]))
    cosLat = np.cos(np.deg2rad(refGeod[0]))
    sinLon = np.sin(np.deg2rad(refGeod[1]))
    cosLon = np.cos(np.deg2rad(refGeod[1]))
    rotMat = np.array([[-sinLon, cosLon, 0],
                       [-sinLat * cosLon, -sinLat * sinLon, cosLat],
                       [cosLat * cosLon, cosLat * sinLon, sinLat]])
    x = rotMat[0,0] * dx + rotMat[0,1] * dy
    y = rotMat[1,0] * dx + rotMat[1,1] * dy + rotMat[1,2] * dz
    z = rotMat[2,0] * dx + rotMat[2,1] * dy + rotMat[2,2] * dz
    return (x, y, z)

def test_ecef_enu():
    pos = (43.21009, 78.120123, 124.0)
    ref = (43.21, 78.12, 120.0)

    print(pos)
    print(geod_2_ecef_pos(pos))
    print(geod_2_ecef_pos_ARCON(pos))
    print(geod_2_enu_pos(pos,ref))

def geoc_2_geod_lat(lat):
    oneMinF = PO_RADIUS_M / EQ_RADIUS_M
    xA = PO_RADIUS_M / np.sqrt(np.tan(np.deg2rad(lat))**2 + oneMinF**2)
    muA = np.arctan(np.sqrt(EQ_RADIUS_M**2 - xA**2) / (oneMinF * xA))
    return np.rad2deg(muA)
    
def units_per_degree(lat, alt=0, units='m'):
    '''Returns a tuple with the number of meters (or designated units) per degree latitude and per
       degree longitude, based on the given latitude and altitude in feet'''
    phi = np.pi / 180 * lat
    if units == 'm':
        a = EQ_RADIUS_M + feet_to_meters(alt)
    elif units == 'ft':
        a = meters_to_feet(EQ_RADIUS_M) + alt
    elif units == 'nmi':
        a = meters_to_nmi(EQ_RADIUS_M) + feet_to_nmi(alt)
    e2 = 0.0066943799901413165
    u = 1 - e2 * np.sin(phi)**2
    M = ((1 - e2) / u) * (a / np.sqrt(u))
    r = np.cos(phi) * (a / np.sqrt(u))
    unitsPerDegLat = (np.pi * M) / 180
    unitsPerDegLon = (np.pi * r) / 180
    return (unitsPerDegLat,unitsPerDegLon)

def latlon_diff(targetGeod, refGeod, units='ft'):
    '''Given a geodetic coordinate tuple(lat, lon, alt; values can be scalars or ndarrays),
       anda a reference coordinate (or set of corresponding reference coordinates)
       returns the north and east range between the coordinates in
       the specified units (ft, m, nmi, or deg)'''
    lat = (targetGeod[0]+refGeod[0])/2
    alt = (targetGeod[2]+refGeod[2])/2
    if (units == 'ft') or (units == 'm') or (units == 'nmi'):
        unitsPerDegLat, unitsPerDegLon = units_per_degree(lat, alt, units)
    elif units == 'deg':
        unitsPerDegLat, unitsPerDegLon = (1,1)
    yDiff = (targetGeod[0] - refGeod[0]) * unitsPerDegLat
    xDiff = (targetGeod[1] - refGeod[1]) * unitsPerDegLon
    return (yDiff, xDiff)

def groundspeed_to_airspeed(groundSpeed,altitude):
    '''Note: approximate
       assumes groundSpeed in knots, altitude in feet'''
    return groundSpeed / (1 + (.02 * (altitude/1000)))

def rate_of_turn(TAS, bankAngle):
    '''TAS: true airspeed in knots
       bankAngle: bank (roll) angle in degrees'''
    return (1091 * np.tan(np.deg2rad(bankAngle))) / TAS
