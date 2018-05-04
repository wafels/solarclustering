#
# A demo of what might be possible
#
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn import metrics
from sklearn.cluster import KMeans

import astropy.units as u
import sunpy.map

from sklearn.decomposition import PCA

# from sunpy.net import Fido, attrs as a

directory = os.path.expanduser('~/Data/solarclustering/demo/1.0/')
#time_range = a.Time('2012/03/04 12:34:00', '2012/03/04 12:34:12')
#for wavelength in (94, 131, 171, 193, 211, 335):
#    result = Fido.search(time_range, a.Instrument('aia'), a.Wavelength(wavelength*u.AA))
#    downloaded_file = Fido.fetch(result[0, 0], path=os.path.join(directory, '{file}.fits'))

np.random.seed(42)
spatial_sum = (2, 2) * u.pix

#
# Define where the files are
#

filenames = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.fits')])

maps = {}
for i, filename in enumerate(filenames):
    # Get the data
    smap = sunpy.map.Map(filename).superpixel(spatial_sum)


    # Set up the joint mask on the firt pass through the loop
    if i == 0:
        joint_mask = np.zeros_like(smap.data, dtype=np.bool)

    # Going to use masks to mask out the bad data
    # First, mask out the non-finites
    non_finite_mask = ~np.isfinite(smap.data)

    # Mask out all the data less than or equal to zero
    zero_or_less_than_mask = smap.data <= 0

    # Combine the masks
    mask = np.logical_or(non_finite_mask, zero_or_less_than_mask)

    # Masked data
    masked_data = np.ma.array(smap.data, mask=mask)

    # Store
    maps[smap.measurement] = sunpy.map.Map(masked_data, smap.meta)

    # Joint mask
    joint_mask = np.logical_or(joint_mask, mask)

    maps[smap.measurement].plot()
    plt.savefig('/home/ireland/Desktop/map_{:s}.png'.format(str(smap.measurement)))


# Normalize the data in each map using the mask in each map
nmaps = {}
for key in maps.keys():
    this_map = maps[key]

    # Transform the data respecting masks
    log_map_data = np.ma.log(this_map.data)

    # Mean of the data respecting masks
    mean = np.ma.mean(log_map_data)

    # Standard deviation of the data respecting masks
    std = np.ma.std(log_map_data)

    # Normalized transformed data
    nmaps[key] = sunpy.map.Map((log_map_data - mean)/std, this_map.meta)

#
# Plot two-dimensional distributions of the normalized
# transformed data using the joint mask.  The joint mask
# indicates where every wavelength has data.
#
keys = list(nmaps.keys())
nkeys = len(keys)
bins = [100, 100]
hrange = [[0, 0], [0, 0]]

# First pass is to get the histogram range so that
# all the plots have the same scaling.
maximum_value = -1
minimum_value = masked_data.size + 1
for i in range(0, nkeys):
    key_i = keys[i]
    skey_i = str(key_i)
    data1 = np.ma.array(nmaps[key_i].data.data, mask=joint_mask).compressed()
    for j in range(i+1, nkeys):
        key_j = keys[j]
        skey_j = str(key_j)
        data2 = np.ma.array(nmaps[key_j].data.data, mask=joint_mask).compressed()

        # Calculate the histogram results
        result = np.histogram2d(data1, data2, bins=bins)

        if result[0].max() > maximum_value:
            maximum_value = result[0].max()

        if result[0].min() < minimum_value:
            minimum_value = result[0].min()

        if result[1].min() < hrange[0][0]:
            hrange[0][0] = result[1].min()

        if result[1].max() > hrange[0][1]:
            hrange[0][1] = result[1].max()

        if result[2].min() < hrange[1][0]:
            hrange[1][0] = result[2].min()

        if result[2].max() > hrange[1][1]:
            hrange[1][1] = result[2].max()


# Second pass is to make the plots with the same color scaling.
for i in range(0, nkeys):
    key_i = keys[i]
    skey_i = str(key_i)
    data1 = np.ma.array(nmaps[key_i].data.data, mask=joint_mask).compressed()
    for j in range(i + 1, nkeys):
        key_j = keys[j]
        skey_j = str(key_j)
        data2 = np.ma.array(nmaps[key_j].data.data, mask=joint_mask).compressed()

        # Do the plot
        plt.close('all')
        plt.hist2d(data1, data2, bins=bins, range=hrange,
                   norm=colors.PowerNorm(0.25, vmin=minimum_value, vmax=maximum_value),
                   cmap='plasma')
        plt.xlabel(key_i)
        plt.ylabel(key_j)
        plt.title('Distribution of intensity\nfull frame, transformed, normalized\n{:s} vs. {:s}'.format(skey_i, skey_j))
        plt.colorbar(label='# pixels ({:n} total)'.format(data2.size))
        plt.clim(minimum_value, maximum_value)
        plt.grid('on', linestyle=":")
        plt.axhline(0.0, color='white', linewidth=0.5)
        plt.axvline(0.0, color='white', linewidth=0.5)
        plt.savefig('/home/ireland/Desktop/fullframe_{:s}_{:s}.png'.format(skey_i, skey_j))

plt.close('all')


"""
#
# Plot two dimensional histograms of the on-disk data only
#
for i in range(0, nkeys):
    # First key
    key_i = keys[i]
    skey_i = str(key_i)

    # First map to look at
    map1 = nmaps[key_i]

    # Calculate a numpy mask indicating where the on disk pixels are
    map1_pixels_on_disk_mask = ???

    for j in range(i+1, nkeys):
        # Second key
        key_j = keys[j]
        skey_j = str(key_j)

        # Second map to look at
        map2 = nmaps[key_j]

        # Calculate a numpy mask indicating where the on disk pixels are
        map2_pixels_on_disk_mask = ???

        # Now calculate a joint mask that shows where the on disk pixels are
        # and the good data for both datasets simultaneously.
        map12_disk_mask = np.logical_or(map1_pixels_on_disk_mask, map2_pixels_on_disk_mask)
        mask = np.logical_or(map12_disk_mask, joint_mask)

        # Get the data
        data1 = np.ma.array(map1.data.data, mask=mask).compressed()
        data2 = np.ma.array(map2.data.data, mask=mask).compressed()

        # Plot
        plt.close('all')
        plt.hist2d(data1, data2, bins=[100, 100])
        plt.xlabel(key_i)
        plt.ylabel(key_j)
        plt.title('Distribution of intensity\ndisk only, transformed, normalized\n{:s} vs. {:s}'.format(skey_i, skey_j))
        plt.colorbar()
        plt.savefig('/home/ireland/Desktop/disk_only_{:s}_{:s}.png'.format(skey_i, skey_j))

plt.close('all')
"""
