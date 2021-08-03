'''
Module for calculating sunrise and sunset times and, more importantly, for the
length of a day (in hours). The core algorithm is `sunrise_sunset()`.
'''

import datetime
import numpy as np

# The sunrise_sunset() algorithm was written for degrees, not radians
_sine = lambda x: np.sin(np.deg2rad(x))
_cosine = lambda x: np.cos(np.deg2rad(x))
_tan = lambda x: np.tan(np.deg2rad(x))
_arcsin = lambda x: np.rad2deg(np.arcsin(x))
_arccos = lambda x: np.rad2deg(np.arccos(x))
_arctan = lambda x: np.rad2deg(np.arctan(x))

def julian_day(year, month, day):
    '''
    Returns the Julian day using the January 1, 2000 epoch. Taken from
    Equation 7.1 in Meeus (1991).

    Parameters
    ----------
    year : int
    month : int
    day : int

    Returns
    -------
    float
    '''
    # Dates in January, February are considered to be on 13th, 14th month
    #   of preceding year
    if month < 3:
        year -= 1
        month += 12
    a = np.floor(year / 100)
    b = 2 - a + np.floor(a / 4)
    return np.floor(365.25 * (year + 4716)) +\
        np.floor(30.6001 * (month + 1)) + day + b - 1524.5


def obliquity(t_solar):
    '''
    Obliquity of the ecliptic as a function of mean solar time; valid only for
    years in 2000 +/- 10,000 years. Taken from Equation 21.2 in Meeus (1991).

    Parameters
    ----------
    t_solar : float
        Mean solar time
    '''
    return 23.43929 - (0.01300416 * t_solar) -\
        (1.638e-7 * np.power(t_solar, 2)) +\
        (5.0361e-7 * np.power(t_solar, 3))


def sunrise_sunset(coords, dt, zenith = -0.83):
    r'''
    Returns the hour of sunrise and sunset for a given date. Hours are on the
    closed interval [0, 23] because Python starts counting at zero; i.e., if
    we want to index an array of hourly data, 23 is the last hour of the day.
    Recommended solar zenith angles for sunrise and sunset are -6 degrees for
    civil sunrise/ sunset; -0.5 degrees for "official" sunrise/sunset; and
    -0.83 degrees to account for the effects of refraction. A zenith angle of
    -0.5 degrees produces results closest to those of pyephem's
    Observer.next_rising() and Observer.next_setting(). This calculation does
    not include corrections for elevation or nutation nor does it explicitly
    correct for atmospheric refraction. Source:

    - U.S. Naval Observatory. "Almanac for Computers." 1990. Reproduced by
      Ed Williams. https://www.edwilliams.org/sunrise_sunset_algorithm.htm

    The algorithm is based on the derivation of the approximate hour angle of
    sunrise and sunset, as described by Jean Meeus (1991) in *Astronomical
    Algorithms*, based on the observer's latitude, `phi`, and the declination
    of the sun, `delta`:

    $$
    \mathrm{cos}(H_0) =
    \frac{\mathrm{sin}(h_0) -
    \mathrm{sin}(\phi)\mathrm{sin}(\delta)}{\mathrm{cos}(\phi)\mathrm{cos}(\delta)}
    $$

    Parameters
    ----------
    coords : list or tuple
        The (latitude, longitude) coordinates of interest; coordinates can
        be scalars or arrays (for times at multiple locations on same date)
    dt : datetime.date
        The date on which sunrise and sunset times are desired
    zenith : float
        The sun zenith angle to use in calculation, i.e., the angle of the
        sun with respect to its highest point in the sky (90 is solar noon)
        (Default: -0.83)

    Returns
    -------
    tuple
        2-element tuple of (sunrise hour, sunset hour)
    '''
    lat, lng = coords
    assert -90 <= lat <= 90, 'Latitude error'
    assert -180 <= lng <= 180, 'Longitude error'
    doy = int(dt.strftime('%j'))
    # Calculate longitude hour (Earth turns 15 degrees longitude per hour)
    lng_hour = lng / 15.0
    # Appoximate transit time (longitudinal average)
    tmean = doy + ((12 - lng_hour) / 24)
    # Solar mean anomaly at rising, setting time
    anomaly = (0.98560028 * tmean) - 3.289
    # Calculate sun's true longitude by calculating the true anomaly
    #   (anomaly + equation of the center), then add (180 + omega)
    #   where omega = 102.634 is the longitude of the perihelion
    lng_sun = (anomaly + (1.916 * _sine(anomaly)) +\
        (0.02 * _sine(2 * anomaly)) + 282.634) % 360
    # Sun's right ascension (by 0.91747 = cosine of Earth's obliquity)
    ra = _arctan(0.91747 * _tan(lng_sun)) % 360
    # Adjust RA to be in the same quadrant as the sun's true longitude, then
    #   convert to hours by dividing by 15 degrees
    ra += np.subtract(
        np.floor(lng_sun / 90) * 90, np.floor(ra / 90) * 90)
    ra_hours = ra / 15
    # Sun's declination's (using 0.39782 = sine of Earth's obliquity)
    #   retained as sine and cosine
    dec_sin = 0.39782 * _sine(lng_sun)
    dec_cos = _cosine(_arcsin(dec_sin))
    # Cosine of the sun's local hour angle
    hour_angle_cos = (
        _sine(zenith) - (dec_sin * _sine(lat))) / (dec_cos * _cosine(lat))
    # Correct for polar summer or winter, i.e., when the sun is always
    #   above or below the horizon
    if hour_angle_cos > 1 or hour_angle_cos < -1:
        if hour_angle_cos > 1:
            return (-1, -1) # Sun is always down
        elif hour_angle_cos < -1:
            return (0, 23) # Sun is always up
    hour_angle = _arccos(hour_angle_cos)
    # Local mean time of rising or setting (converting hour angle to hours)
    hour_rise = ((360 - hour_angle) / 15) + ra_hours -\
    (0.06571 * (tmean - 0.25)) - 6.622
    hour_sets = (hour_angle / 15) + ra_hours -\
    (0.06571 * (tmean + 0.25)) - 6.622
    # Round to nearest hour, convert to UTC
    return (
        np.floor((hour_rise - lng_hour) % 24),
        np.floor((hour_sets - lng_hour) % 24))
