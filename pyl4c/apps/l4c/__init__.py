'''
'''

import csv
import datetime
import warnings
import numpy as np

def report(hdf):
    '''
    Check that we have everything needed to run L4C, print a report to the
    screen.

    Parameters
    ----------
    hdf : h5py.File
    '''
    KEYS = ('apar', 'vpd', 'ft', 'tmin', 'tsoil', 'smrz', 'smsf')

    def find(hdf, prefix, key, pad = 10):
        'Find a key, print the report'
        try:
            field = '%s/%s' % (prefix, key)
            if len(hdf[field].shape) == 2 or key == 'fpar':
                pretty = ('"%s"' % key).ljust(pad)
                print_stats(hdf[field][:], pad, pretty)
            elif len(hdf[field].shape) == 3:
                # Assuming data are enumerated on the first axis
                for i in range(0, hdf[field].shape[0]):
                    pretty = ('"%s" (%d)' % (key, i)).ljust(pad)
                    print_stats(hdf[field][i,...], pad, pretty)
        except KeyError:
            pretty = ('"%s"' % key).ljust(pad)
            print('-- MISSING %s' % pretty)

    def print_stats(data, pad, pretty):
        shp = ' x '.join(map(str, data.shape))
        shp = ('[%s]' % shp).ljust(pad + 7)
        stats = tuple(summarize(data))
        stats_pretty = ''
        if stats[0] is not None:
            stats_pretty = '[%.2f, %.2f]' % (stats[0], stats[2])
            if len(key) < 10:
                print('-- Found %s %s %s' % (pretty, shp, stats_pretty))
            else:
                print('-- Found %s' % pretty)
                print('%s%s %s' % (''.rjust(pad + 10), shp, stats_pretty))

    def summarize(data, nodata = -9999):
        'Get summary statistics for a field'
        if str(data.dtype).startswith('int'):
            return (None for i in range(0, 3))
        if data.dtype in (np.float32, np.float64):
            data[data == -9999] = np.nan
        return (
            getattr(np, f)(data) for f in ('nanmin', 'nanmean', 'nanmax')
        )

    print('\nL4C: Validating configuration and input datasets for file:')
    print('  %s' % hdf.filename)
    print('\nL4C: Checking for required driver variables...')
    for key in KEYS:
        if key == 'ft' and key not in hdf['drivers'].keys():
            find(hdf, 'drivers', 'tsurf')
        elif key == 'apar' and key not in hdf['drivers'].keys():
            find(hdf, 'drivers', 'par')
            find(hdf, 'drivers', 'fpar')
        else:
            find(hdf, 'drivers', key)

    print('\nL4C: Checking for required state variables...')
    for key in ('PFT', 'npp_sum', 'soil_organic_carbon',):
        find(hdf, 'state', key)

    print('\nL4C: Summarizing metadata...')
    y1, m1, d1, _ = hdf['time'][0,...]
    y2, m2, d2, _ = hdf['time'][-1,...]
    print('-- First date: %s' % datetime.datetime(y1, m1, d1)\
        .strftime('%Y-%m-%d'))
    print('-- Final date: %s' % datetime.datetime(y2, m2, d2)\
        .strftime('%Y-%m-%d'))
    print('-- Total length: %d' % hdf['time'].shape[0])
    print('')
