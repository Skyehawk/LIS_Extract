# LIS Data Extract (Plot off) - v0.20
# Python Version: 3.7.3
#
# Skye Leake
# 2019-11-27
#
# Updated
# 2020-07-05
#
# Developed as a tool to extract values from a grb file for overlay geostatistical analysis
# 
# Use:	Linux call: python /mnt/d/Libraries/Documents/Scripts/LIS_Plot/LIS_Extract_Temps_Plot.py -l /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/SPoRT_LIS/SPoRT_LIS/2016/sportlis_daily_forSkye_20160715/201607/LIS_HIST_201607150000.d01.grb -g /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/GFS/2016/gfsanl_4_20160715_0600_000.grb2 -o /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/Output/2016/20160715_test -c 42.087 -102.882
#		
#
# Notes: Output currently .csv & ascii (in .txt format) files

# --- Imports ---
import os
import argparse
import numpy as np
import datetime
import pygrib
import matplotlib.pyplot as plt						# v. 3.1.0
import matplotlib.colors as colors
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
from skimage.transform import rotate				# v. 0.15.0

from Transformation_Matrix_2 import comp_matrix
from Gauss_Map import gauss_map 

# Debug (progress bars bugged by matplotlib futurewarnings output being annoying)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Construct argument parse to parse the arguments (input dates) ---
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--GFS", help=" path to the input file of GFS Wind U*V data (GRIB)")
ap.add_argument("-o", "--output", help=" path to the output directory")
ap.add_argument("-c", "--lat_lon", nargs="+", help="passed: -c <lat_float> <lon_float>; Lat/Lon of point of convection")
args = vars(ap.parse_args())
outputPath = args["output"]

GFSGrbs = pygrib.open(args["GFS"])
testGrb = GFSGrbs.select()[0]
out = []

levels = [950,900,850,800,750]
for level in levels:
	UGrb = GFSGrbs.select(name='U component of wind', typeOfLevel='isobaricInhPa', level=level)[0]
	print("\nU " + str(level) + "hPa Data: " + str(UGrb))
	U = UGrb.values * 1.944

	VGrb = GFSGrbs.select(name='V component of wind', typeOfLevel='isobaricInhPa', level=level)[0]
	print("\nV " + str(level) + "hPa Data: " + str(VGrb))
	V = VGrb.values * 1.944	

	ULons = np.linspace(float(UGrb['longitudeOfFirstGridPointInDegrees']), float(UGrb['longitudeOfLastGridPointInDegrees']), int(UGrb['Ni']) )
	ULats = np.linspace(float(UGrb['latitudeOfFirstGridPointInDegrees']), float(UGrb['latitudeOfLastGridPointInDegrees']), int(UGrb['Nj']) )	
	U, ULons = shiftgrid(180., U, ULons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
	UGridLon, UGridLat = np.meshgrid(ULons, ULats) 

	VLons = np.linspace(float(VGrb['longitudeOfFirstGridPointInDegrees']), float(VGrb['longitudeOfLastGridPointInDegrees']), int(VGrb['Ni']) )	
	VLats = np.linspace(float(VGrb['latitudeOfFirstGridPointInDegrees']), float(VGrb['latitudeOfLastGridPointInDegrees']), int(VGrb['Nj']) )	
	V, VLons = shiftgrid(180., V, VLons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
	#VGridLon, VGridLat = np.meshgrid(VLons, VLats) 							#disabled

	testLoc = np.array([float(args["lat_lon"][1]),float(args["lat_lon"][0])])
	gridTestLoc = np.around(testLoc*2)/2										# round to the nearest 1.0 for GFS3, 0.5 for GFS4
	testLocIdx= np.where(gridTestLoc[1]==UGridLat)[0][0], np.where(gridTestLoc[0]==UGridLon)[1][0]
	testValU = U[testLocIdx[0]-1: testLocIdx[0] + 2,testLocIdx[1]-1: testLocIdx[1] + 2]			# 3x3 sample of U values centered about our closest vector
	testValV = V[testLocIdx[0]-1: testLocIdx[0] + 2,testLocIdx[1]-1: testLocIdx[1] + 2]			# 3x3 sample of V values centered about our closest vector
	gausKern3x3sig1 = np.array([[0.077847,0.123317,0.077847],\
								[0.123317,0.195346,0.123317],\
								[0.077847,0.123317,0.077847]])
	testLocBearing = np.arctan2(np.sum(testValV*gausKern3x3sig1), np.sum(testValU*gausKern3x3sig1))
	testLocMag = np.sqrt(np.sum(testValV*gausKern3x3sig1)**2 + np.sum(testValU*gausKern3x3sig1)**2)


	# We need to put a flag in here of the variance is over a certian level (we could use Levene test, and split on... what?)(or include the variance straight up and deal with it later(weighted or not?))

	out.append([level, testLocBearing, testLocMag])

outA = np.array(out)
print(out)
f_o = open(args["output"] + 'log_stats_area.txt', 'a')
f_o.write(str(datetime.datetime.strptime(str(testGrb.dataDate), '%Y%m%d')) 
	+ '\t' + str(args["lat_lon"]) 
	+ '\t' + str(outA[0,1]) + '\t' + str(outA[0,2])				# 950hPa, Bearing/Mag
	+ '\t' + str(outA[1,1]) + '\t' + str(outA[1,2])				# 900hPa, Bearing/Mag
	+ '\t' + str(outA[2,1]) + '\t' + str(outA[2,2])				# 850hPa, Bearing/Mag
	+ '\t' + str(outA[3,1]) + '\t' + str(outA[3,2])				# 800hPa, Bearing/Mag
	+ '\t' + str(outA[4,1]) + '\t' + str(outA[4,2]) + '\n')		# 750hPa, Bearing/Mag
f_o.close()