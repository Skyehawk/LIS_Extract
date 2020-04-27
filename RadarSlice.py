import warnings

import numpy as np
from datetime import datetime
from metpy.io import Level3File

from Transformation_Matrix_2 import comp_matrix

class RadarSlice(object):

    @property
    def datetime(self):
        if not hasattr(self,'_datetime'):
            self._datetime = None
            warnings.warn("Radar_Slice: No input datetime parsed on __init__",  UserWarning)
        return self._datetime

    @property
    def data(self):
        if not hasattr(self,'_data'):
            self._data = None
            warnings.warn("Radar_Slice: No input data parsed on __init__", UserWarning)
        return self._data

    @property
    def sensorLocation(self):
        if not hasattr(self,'_sensorLocation'):
            self._sensorLocation = None
            warnings.warn("Radar_Slice: No input sensor location parsed on __init__", UserWarning)
        return self._sensorLocation

    @property
    def xlocs(self):
        return self._xlocs

    @property
    def ylocs(self):
        return self._ylocs

    @property
    def area(self):
        if not hasatr(self, '_area'):
            self._area = 0.0
        return self._area

    @property
    def meanReflectivity(self):
        if not hasatr(self, '_meanReflectivity'):
            self._meanReflectivity = 0.0
        return self._meanReflectivity

    @datetime.setter
    def datetime(self, value):
        self._datetime = value

    @data.setter
    def data(self, value):
        self._data = value

    @sensorLocation.setter
    def sensorLocation(self, value):
        self._sensorLocation = value

    @xlocs.setter
    def xlocs(self, value):
        self._xlocs = value

    @ylocs.setter
    def ylocs(self, value):
        self._ylocs = value

    @area.setter
    def area(self,value):
        self._area = value

    @meanReflectivity.setter
    def meanReflectivity(self,value):
        self._meanReflectivity = value


    # Constructor (init)
    def __init__(self, file, sensorLocation=np.array([0.0,0.0])):
        self.file = file
        f = Level3File(self.file)
        dataDict = f.sym_block[0][0]                                        # Pull the data dictionary out of the file object        
        self.datetime = f.metadata['prod_time']
        self.data = np.ma.array(dataDict['data'])                           # Turn into an array, then mask
        self.sensorLocation = sensorLocation

        self.az = np.array(dataDict['start_az'] + [dataDict['end_az'][-1]]) # Grab azimuths and calculate a range based on number of gates
        self.rng = rng = np.linspace(0, f.max_range, self.data.shape[-1] + 1)

    def calc_cartesian(self):
        self.kmPerDeg = 111.0
        self.xlocs = (self.rng[:-1] * np.sin(np.deg2rad(self.az[1:, None]))/self.kmPerDeg) # Convert az,range to x,y, change to deg from km
        self.ylocs = (self.rng[:-1] * np.cos(np.deg2rad(self.az[1:, None]))/self.kmPerDeg)
        return(True)

    def shift_cart_orgin(self, offset):
        #offset must be in (lat,lon)
        self.xlocs += offset[1]
        self.ylocs += offset[0]

        if self.ylocs.any() > 90 or self.ylocs.any() < -90:
            warnings.warn("Radar_Slice: offset (lat) produces values out of expected range [-90:90]",  UserWarning)
        if self.xlocs.any() > 180 or self.xlocs.any() < -180:
            warnings.warn("Radar_Slice: offset (lon) produces values out of expected range [-180:180]",  UserWarning)

        return True

    def mask_zeros(self):
        self.data[self.data == 0] = np.ma.masked                             # convert 0s to masked
        return True

    def find_area(self, reflectThresh=0.0):
        self.area = sum(map(lambda i: i >= reflectThresh, self.data.flatten()))
        return self.area

    def find_mean_reflectivity(self,reflectThresh=0.0):
        self.meanReflectivity = np.mean(np.array(list(filter(lambda x: x >= reflectThresh, self.data.flatten()))))
        return self.meanReflectivity

    def __str__(self):
        return ('string method for radarSlice')