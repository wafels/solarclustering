#
# A demo of what might be possible
#
import os

class ObservationFileInfo:
    def __init__(self, directory, filename, observation=None):
        self.directory = directory
        self.filename = filename
        self.level = None
        if observation is not None:
            self.observation = self._get_observation_from_fileinfo()

        self.filepath = os.path.join(self.directory, self.filename)


    def _get_observation(self):
        """
        Finds out what kind of observation is encoded by the filename.
        :param filename:
        :return:
        """
        pass

#
# Define where the files are
#
directory = os.path.expanduser('~/Data/solarclustering')
filenames = [f for f in listdir(directory) if isfile(join(directory, f))]
these_files = []
for filename in filenames:
    these_files.append(ObservationFileInfo(directory, filename))


#
# Read file files and extract the data we need
#
#for f in these_files:
#    pass

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