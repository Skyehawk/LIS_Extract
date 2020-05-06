# NEXRAD Data Encapsulation - v0.1.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-04-28
#
# Updated
# 2020-04-28
#

# --- Imports ---
import numpy as np
from matplotlib.path import Path
from Transformation_Matrix_2 import comp_matrix
from RadarSlice import RadarSlice

class RadarROI(RadarSlice):
    
    @property
    def clippedData(self):
        if not hasattr(self, '_clippedData'):
            self._clippedData = np.empty((1))
        return self._clippedData

    @property
    def mask(self):
        return self._mask
    
    @property
    def area(self):
        if not hasattr(self, '_area'):
            self._area = 0.0
        return self._area

    @property
    def meanReflectivity(self):
        if not hasattr(self, '_meanReflectivity'):
            self._meanReflectivity = 0.0
        return self._meanReflectivity

    @property
    def varReflectivity(self):
        if not hasattr(self, '_varReflectivity'):
            self._varReflectivity = 0.0
        return self._varReflectivity

    @property
    def polyVerts(self):
        if not hasattr(self, '_polyVerts'):
            self._polyVerts = []
        return self._polyVerts
    
    @property
    def tm(self):
        if not hasattr(self,'_tm'):
            self._tm = comp_matrix(scale=np.ones(2), rotation=np.zeros(2), 
                                    shear=np.ones(2), translation=np.zeros(2))
        return self._tm

    @clippedData.setter
    def clippedData(self, value):
        self._clippedData = value

    @mask.setter
    def mask(self, value):
        self._mask = value

    @area.setter
    def area(self,value):
        self._area = value

    @meanReflectivity.setter
    def meanReflectivity(self, value):
        self._meanReflectivity = value

    @varReflectivity.setter
    def varReflectivity(self, value):
        self._varReflectivity = value

    @polyVerts.setter
    def polyVerts(self,value):
        self._polyVerts = value

    @tm.setter
    def tm(self, value):
        self._tm = value

    #Override
    def __init__(self, file, sensorLocation):
        super(RadarROI, self).__init__(file, sensorLocation)

    def extractROI(self, baseCrds=None, baseBearing=0.0):
        if baseCrds is None:
            baseCrds = np.array([(1.0,1.0,0.0,1.0),
                        (1.0,-1.0,0.0,1.0),
                        (-1.0,-1.0,0.0,1.0),
                        (-1.0,1.0,0.0,1.0),
                        (1.0,1.0,0.0,1.0)])    #crds of bounding box (Gridded degrees)
        
        self.tm = comp_matrix(scale=np.ones(3), rotation=np.array([0,0, baseBearing]), 
                        shear=np.ones(3), translation=np.zeros(3))

        self.polyVerts = self.tm.dot(baseCrds.T).T[:,:2]    # Apply transformation Matrix, remove padding, and re-transpose
        
        # --- Generate ROI from coordiantes (above) create 2D boolean array to mask with ---
        xp,yp = self.xlocs.flatten(),self.ylocs.flatten()
        points = np.vstack((xp,yp)).T
        path = Path(self.polyVerts)
        grid = path.contains_points(points)
        grid = grid.reshape(np.shape(self.xlocs))
        rDataMasked = np.ma.masked_array(self.data, np.invert(grid))

        # --- Clip our masked array, create sub-array of data and rotate ---
        i, j = np.where(grid)
        self.mask = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')
        rDataMaskedClip = self.data[self.mask]
        rDataMaskClip = grid[self.mask]
        self.clippedData = rDataMaskedClip*rDataMaskClip
        self.xlocs = self.xlocs[self.mask]
        self.ylocs = self.ylocs[self.mask]
        return self.clippedData

    #Override
    def find_area(self, reflectThresh=0.0):
        self.area = sum(map(lambda i: i >= reflectThresh, self.clippedData.flatten()))
        return self.area

    #Override
    def find_mean_reflectivity(self, reflectThresh=0.0):
        self.meanReflectivity = np.mean(np.array(list(filter(lambda x: x >= reflectThresh, self.clippedData.flatten()))))
        return self.meanReflectivity

    #Override
    def find_variance_reflectivity(self, reflectThresh=0.0):
        self.varReflectivity = np.var(np.array(list(filter(lambda x: x >= reflectThresh, self.clippedData.flatten()))))
        return self.varReflectivity

    #Override
    def __str__(self):
        return ('string method for radarRoi')