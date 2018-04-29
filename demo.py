#
# A demo of what might be possible
#
import os
import glob

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

import astropy.units as u
import sunpy.map


def data_simple_replace_zero_values(data, replacement_value=0.001):
    """
    Replace zero values in a numpy array with a fixed replacement value.
    :param data:
    :param replacement_value:
    :return:
    """
    return data_simple_replace(data, data == 0, replacement_value)


def data_simple_replace_negative_values(data, replacement_value=0.001):
    """
    Replace negative values in a numpy array with a fixed replacement value.
    :param data:
    :param replacement_value:
    :return:
    """
    return data_simple_replace(data, data < 0, replacement_value)


def data_simple_replace_nans(data, replacement_value=0.001):
    """
    Replace NaNs in a numpy array with a fixed replacement value.
    :param data:
    :param replacement_value:
    :return:
    """
    return data_simple_replace(data, ~np.isfinite(data), replacement_value)


def data_simple_replace(data, condition, replacement_value):
    """
    Replace values in a numpy array with the replacement value where the input
    condition is True and return a new array.
    :param data:
    :param condition:
    :param replacement_value:
    :return:
    """
    newdata = deepcopy(data)
    newdata[condition] = replacement_value
    return newdata


"""
class ObservationFileInfo:
    def __init__(self, directory, filename, observation=None):
        self.directory = directory
        self.filename = filename
        self.level = None
        if observation is not None:
            self.observation = self._get_observation_from_fileinfo()

        self.filepath = os.path.join(self.directory, self.filename)
"""


#    def _get_observation(self):
#        """
#        Finds out what kind of observation is encoded by the filename.
#        :param filename:
#        :return:
#        """
#        pass

spatial_sum = 4
n_image_pixels = (4096/spatial_sum) ** 2
dimensionality = 6

#
# Define where the files are
#

directory = os.path.expanduser('~/Data/solarclustering/demo')
filenames = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.fits')])
all_data = np.zeroes((n_image_pixels, dimensionality))
for filename in filenames:
    # Get the data
    data = sunpy.map.Map(filename).superpixel(spatial_sum, spatial_sum).data

    # Clean the data
    data = data_simple_replace_nans(data, replacement_value=1.0)
    data = data_simple_replace_zero_values(data, replacement_value=1.0)
    data = data_simple_replace_negative_values(data, replacement_value=1.0)

    # Log the data
    data = np.log(data)

    # Central measurement
    #data_center = np.median(data)

    # Width measurement
    #data_width = mad(data)

    # Normalize
    #data = (data - data_center) / data_width

    data = scale(data)

    # Store
    these_files.append(data.flatten())


#
# Clean up and normalize the data
# Find where the negatives in each dataset and mask them
#
# How best to normalize the data?.
# Take log, then define a width measure, either the standard deviation, or
# median absolute deviation?
# Then remove mean, mode or median and divide by measurement of width
#
#


#
# Do the cluster analysis
#