# NEXRAD Data Encapsulation - v0.1.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-04-27
#
# Updated
# 2020-04-28
#

# --- Imports ---
import warnings

import numpy as np
from datetime import datetime
from metpy.io import Level3File

from Transformation_Matrix_2 import comp_matrix

class RadarSlice(object):

    #Full list of properties avaliable at: https://unidata.github.io/MetPy/latest/api/generated/metpy.io.Level3File.html

    @property
    def metadata(self):
        if not hasattr(self,'_metadata'):
            self._metadata = None
            warnings.warn("Radar_Slice: No input metadata parsed on __init__", UserWarning)
        return self._metadata
    
    @property
    def datetime(self):
        if not hasattr(self,'_datetime'):
            self._datetime = None
            warnings.warn("Radar_Slice: No input datetime parsed on __init__ , check input metadata",  UserWarning)
        return self._datetime

    @property
    def data(self):
        if not hasattr(self,'_data'):
            self._data = None
            warnings.warn("Radar_Slice: No input data parsed on __init__", UserWarning)
        return self._data

    @property
    def sensorData(self):
        if not hasattr(self,'_sensorData'):
            self._sensorData = None
            warnings.warn("Radar_Slice: No input sensor metadata parsed on __init__", UserWarning)
        return self._sensorData

    @property
    def mapData(self):
        if not hasattr(self,'_mapData'):
            self._mapData = None
            warnings.warn("Radar_Slice: No input sensor data mapping values parsed on __init__", UserWarning)
        return self._mapData
    
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

    @property
    def varReflectivity(self):
        if not hasattr(self, '_varReflectivity'):
            self._varReflectivity = 0.0
        return self._varReflectivity

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @datetime.setter
    def datetime(self, value):
        self._datetime = value

    @data.setter
    def data(self, value):
        self._data = value

    @sensorData.setter
    def sensorData(self, value):
        self._sensorData = value

    @mapData.setter
    def mapData(self, value):
        self._mapData = value

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

    @varReflectivity.setter
    def varReflectivity(self, value):
        self._varReflectivity = value

    # Constructor (init)
    def __init__(self, file, sensorData=None):
        self.file = file
        f = Level3File(self.file)
        dataDict = f.sym_block[0][0]                                        # Pull the data dictionary out of the file object        
        
        self.metadata = f.metadata
        self.datetime = f.metadata['prod_time']                             # More convienent access to the production time datetime object
        self.data = np.ma.array(dataDict['data'])                           # Turn into an array, then mask
        if sensorData is not None:
            self.sensorData = sensorData
        else:
            self.sensorData = {'siteID':f.siteID,'lat': f.lat,'lon': f.lon, 'height':f.height}
        self.mapData = f.map_data

        self.az = np.array(dataDict['start_az'] + [dataDict['end_az'][-1]]) # Grab azimuths and calculate a range based on number of gates
        self.rng = rng = np.linspace(0, f.max_range, self.data.shape[-1] + 1)

    def calc_cartesian(self):
        self.kmPerDeg = 111.0
        self.xlocs = (self.rng[:-1] * np.sin(np.deg2rad(self.az[1:, None]))/self.kmPerDeg) # Convert az,range to x,y, change to deg from km
        self.ylocs = (self.rng[:-1] * np.cos(np.deg2rad(self.az[1:, None]))/self.kmPerDeg)
        return True

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

    def find_mean_reflectivity(self, reflectThresh=0.0):
        self.meanReflectivity = np.mean(np.array(list(filter(lambda x: x >= reflectThresh, self.data.flatten()))))
        return self.meanReflectivity

    def find_variance_reflectivity(self, reflectThresh=0.0):
        self.varReflectivity = np.var(np.array(list(filter(lambda x: x >= reflectThresh, self.clippedData.flatten()))))
        return self.varReflectivity

    def __str__(self):
        return ('string method for radarSlice')