#
# Initial look
#
# Take a short time range, and take a look
# at the data in that short time range.
# Assume one file for each wavelength inside that
# time range.
# Plot 2d histograms of the normalized scaled intensity
# of each EUV wavelength against each other.
#
# To look at more extended time ranges, will need to
# (1) calculate joint masks for every sextuple of images
# (2) extract the intensities from the data
#
#
import os
import datetime

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.units as u

import sunpy.map
from sunpy.instr.aia import aiaprep
from sunpy.net import Fido, attrs as a
from sunpy.time import parse_time

# Where the data is stored
data_root = os.path.expanduser('~/Data/solarclustering/demo/')

# String formatting for time
string_format_time = "%Y%m%d_%H%M%S"

# Start times
start_times = ('2012/03/04 12:34:00', '2012/03/01 03:30:00')

# Wavelengths to look at
wavelengths = (94, 131, 171, 193, 211, 335, 304, 1600, 1700)

# Search time range around the start time
search_time_range = 24 * u.s

# Spatial summing
spatial_sum = (2, 2) * u.pix

# Storage for all the filepaths
all_files = dict()

# Get all the data
for start_time in start_times:
    #
    all_files[start_time] = dict()

    # Get the start time for the search
    st = parse_time(start_time)

    # Get the end time for the search
    et = st + datetime.timedelta(seconds=search_time_range.to(u.s).value)

    # Create the search time range
    time_range = a.Time(st, et)

    # Create a string that will be used to name the subdirectory
    start_time_string = st.strftime(string_format_time)

    # Create the subdirectory of the data root
    sub_data_root = os.path.join(data_root, start_time_string)

    # Create and store Level 1.0 and 1.5 subdirectories
    level_sub_dirs = {}
    for level in ('1.0', '1.5'):
        all_files[start_time][level] = []
        level_sub_dir = os.path.join(sub_data_root, level)
        level_sub_dirs[level] = level_sub_dir
        if not os.path.exists(level_sub_dir):
            os.makedirs(level_sub_dir)

    # Download all data
    lsd = level_sub_dirs["1.0"]
    for wavelength in wavelengths:
        # Search for the files at this wavelength
        result = Fido.search(time_range, a.Instrument('aia'), a.Wavelength(wavelength*u.AA))

        # Download the first file found
        downloaded_file = Fido.fetch(result[0, 0], path=os.path.join(lsd, '{file}.fits'))

        # Store this set of filepaths
        all_files[start_time]["1.0"].append(downloaded_file[0])

    # Promote to Level 1.5
    filepaths = [os.path.join(lsd, f) for f in os.listdir(lsd) if f.endswith('.fits')]
    for fp in filepaths:
        # Get the filename out of filepath
        dummy, filename = os.path.split(fp)

        # Filepath for the Level 1.5 data
        fp15 = os.path.join(level_sub_dirs["1.5"], "{:s}{:s}".format(filename, '.1.5.fits'))

        # Promote to Level 1.5
        promoted = aiaprep(sunpy.map.Map(fp))

        # Save the Level 1.5 file
        promoted.save(fp15)

        # Store this set of filepaths
        all_files[start_time]["1.5"].append(fp15)

# Number of filesets
n_fileset = len(all_filesets)

# Storage for the compressed data
compressed = {}

# Go through each set of files and get the compressed data
for j, filepaths15 in enumerate(all_filesets):
    print('File set {:n}'.format(j))
    print(filepaths15)

    # Initialize the maps for this set of data.
    maps = {}

    # Load in each file and get the mask
    for i, filename in enumerate(filepaths15):
        # Get the image data
        smap = sunpy.map.Map(filename).superpixel(spatial_sum)

        # Size of each source image
        image_size = smap.data.size

        # Set up the joint mask on the first pass through the loop.
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

        # Masked map
        maps[smap.measurement] = sunpy.map.Map(masked_data, smap.meta)

        # Joint mask
        joint_mask = np.logical_or(joint_mask, mask)

        #maps[smap.measurement].plot()
        #plt.savefig('/home/ireland/Desktop/map_{:s}.png'.format(str(smap.measurement)))

    # Apply the joint mask to each wavelength in the set.
    for key in maps.keys():
        this_data = maps[key].data.data
        # Get the unmasked data only
        compressed_data = np.ma.array(this_data, mask=joint_mask).compressed()

        # If this is the first time we have seen this key then add it to the dictionary
        # Otherwise, concatenate the existing and new data
        if key not in list(compressed.keys()):
            compressed[key] = compressed_data
        else:
            compressed[key] = np.concatenate((compressed[key], compressed_data))


def transform_then_normalize(d, transform=np.log, center=np.mean, width=np.std):
    """
    Convenience function to transform the input data and
    :param d:
    :return:
    """
    dt = transform(d)
    dcenter = center(dt)
    dwidth = width(dt)
    return (dt - dcenter)/dwidth


#
# Plot two-dimensional distributions of the normalized
# transformed data using the joint mask.  The joint mask
# indicates where every wavelength has data.
#
keys = list(compressed.keys())
nkeys = len(keys)
bins = [100, 100]
hrange = [[0, 0], [0, 0]]

# First pass is to get the histogram range so that
# all the plots have the same scaling.
maximum_value = -1
minimum_value = compressed[keys[0]].size + 1
for i in range(0, nkeys):
    key_i = keys[i]
    skey_i = str(key_i)
    data1 = transform_then_normalize(compressed[key_i])
    for j in range(i+1, nkeys):
        key_j = keys[j]
        skey_j = str(key_j)
        data2 = transform_then_normalize(compressed[key_j])

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
    data1 = transform_then_normalize(compressed[key_i])
    for j in range(i + 1, nkeys):
        key_j = keys[j]
        skey_j = str(key_j)
        data2 = transform_then_normalize(compressed[key_j])

        # Do the plot
        plt.close('all')
        plt.hist2d(data1, data2, bins=bins, range=hrange,
                   norm=colors.PowerNorm(0.25, vmin=minimum_value, vmax=maximum_value),
                   cmap='plasma')
        plt.xlabel(key_i)
        plt.ylabel(key_j)
        plt.title('Distribution of intensity\nfull frame, transformed, normalized\n{:s} vs. {:s}\n{:n} image sets'.format(skey_i, skey_j, n_fileset))
        plt.colorbar(label='# pixels ({:n} good pixels)\n{:n}% used of all possible'.format(data2.size, 100*data2.size/(n_fileset*image_size)))
        plt.clim(minimum_value, maximum_value)
        plt.grid('on', linestyle=":")
        plt.axhline(0.0, color='white', linewidth=0.5)
        plt.axvline(0.0, color='white', linewidth=0.5)
        plt.tight_layout()
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
