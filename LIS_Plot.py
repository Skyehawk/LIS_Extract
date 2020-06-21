# LIS Data Viewer - v0.01
# Python Version: 3.7.3
#
# Skye Leake
# 08-30-2019
#
# Developed as a tool to view SPoRT Land Information Systam (LIS) data for MCS Research, WMU geography thesis work
# 
# Use:
#
# Notes:

import argparse

import numpy as np
#import glo											# v0.7	
#import PIL.Image as Image							# v1.1.7
import matplotlib.pyplot as plt						# v3.1.0
import matplotlib.colors as colors
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
#import matplotlib.image as mpimg
#import cartopy										# v0.17.0
#import pandas as pd 								# v0.25.1
import pygrib

#-------------
# Construct argument parse to parse the arguments (input dates)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help=" path to the input file (GRIB)")
ap.add_argument("-o", "--output", help="(Optional) override path to the output directory")
args = vars(ap.parse_args())

inputPath = args["input"]
outputPath = args["output"]

grib = args["input"] # Set the file name of your input GRIB file
grbs = pygrib.open(grib)

#grb = grbs.select()[32] #RSM 0-10 in LIS (9 is VSM0-10) (for dates 2015 & after, before this it is idx position 15)
grb = grbs.select()[15]
print("\nData: " + str(grb))
data = grb.values

print("Data Cell Count:"+ "\nNi: " + str(float(grb['Ni'])) + "\nNj: " + str(float(grb['Nj'])))
print("Data Range: " + "\nLon: " + str(float(grb['longitudeOfFirstGridPointInDegrees'])) + " --> " +
		str(float(grb['longitudeOfLastGridPointInDegrees'])) + "\nLat: " + 
		str(float(grb['latitudeOfFirstGridPointInDegrees'])) + " --> " + 
		str(float(grb['latitudeOfLastGridPointInDegrees'])))

lons = np.linspace(float(grb['longitudeOfFirstGridPointInDegrees']), 
		float(grb['longitudeOfLastGridPointInDegrees']), int(grb['Ni']) )	#generate x coordinates normalized to the range of data
lats = np.linspace(float(grb['latitudeOfFirstGridPointInDegrees']), 
		float(grb['latitudeOfLastGridPointInDegrees']), int(grb['Nj']) )	#generate y coordinates normalized to the range of data

# need to shift data grid longitudes from (0..360) to (-180..180)? (do not use for LIS, do not use for GFS)
# data, lons = shiftgrid(180., data, lons, start=False)  # use this to bump data from 0..360 on the lons to -180..180
grid_lon, grid_lat = np.meshgrid(lons, lats) #regularly spaced 2D grid

print("Adj Lon Range: " + str(grid_lon[0,0]) + " --> " + str(grid_lon[0,-1]))
#print("Data test: " , data[-102.210999:43.824593,40:50].T)

plt.figure(figsize=(12,8))
m = Basemap( projection='lcc', resolution='c', rsphere=(6378137.00,6356752.3142), 
			llcrnrlon=-108.0, urcrnrlon=-92.50, llcrnrlat=38.0, 
			urcrnrlat=50., lat_1=33., lat_2=45, lat_0=39, lon_0=-96.)

x, y = m(grid_lon, grid_lat)

cs = m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.gist_stern_r)

#m.readshapefile('/mnt/d/Libraries/Documents/Scripts/LIS_Plot/Test_Data/Aux_files/States_21basic/states', 'states')
#m.readshapefile('/mnt/d/Libraries/Documents/Scripts/LIS_Plot/Test_Data/Aux_files/Canada/Canada', 'Canada')
m.drawcoastlines()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,30.),labels=[0,0,0,1])

plt.colorbar(cs,orientation='vertical', shrink=0.5)
plt.title('SPoRT LIS: RSM 0-10cm') # Set the name of the variable to plot
plt.savefig(grib +'.png') # Set the output file name