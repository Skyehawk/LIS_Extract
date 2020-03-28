# NEXRAD Data Extract - v0.10
# Python Version: 3.7.3
#
# Skye Leake
# 2020-03-27
#
# Updated
# 2020-03-28
#
# Developed as a tool to extract values from NEXRAD inputs for Thesis work
# 
# Use:	Linux call: python /mnt/d/Libraries/Documents/Scripts/LIS_Plot/LIS_Extract.py -l /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/SPoRT_LIS/SPoRT_LIS/2016/sportlis_daily_forSkye_20160715/201607/LIS_HIST_201607150000.d01.grb -g /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/GFS/2016/gfsanl_4_20160715_0600_000.grb2 -o /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/Output/2016/20160715_test -c 42.087 -102.882
#		
#
# Notes: Output currently .csv & ascii (in .txt format) files

# --- Imports ---
import os
import argparse
import numpy as np
import pygrib
import matplotlib.pyplot as plt						# v. 3.1.0
import matplotlib.colors as colors
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
from skimage.transform import rotate				# v. 0.15.0
#from metpy.cbook import get_test_data
from metpy.io import Level3File
from metpy.plots import add_timestamp, colortables

from Transformation_Matrix_2 import comp_matrix
from Gauss_Map import gauss_map 

#  --- Construct argument parse to parse the arguments (input dates) ---
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--NEXRADL3", help=" path to the input file of NEXRADL3 Data ()")
ap.add_argument("-g", "--GFS", help=" path to the input file of GFS Wind U*V data (GRIB)")
ap.add_argument("-o", "--output", help=" path to the output directory")
ap.add_argument("-c", "--lat_lon", nargs="+", help="passed: -c <lat_float> <lon_float>; Lat/Lon of point of convection")
args = vars(ap.parse_args())
outputPath = args["output"]

# --- test ---
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
for v, ctable, ax in zip(('N0Q', 'N0U'), ('NWSReflectivity', 'NWSVelocity'), axes):
	# Open the file
	f = Level3File(args["NEXRADL3"])

	# Pull the data out of the file object
	datadict = f.sym_block[0][0]

	# Turn into an array, then mask
	data = np.ma.array(datadict['data'])
	data[data == 0] = np.ma.masked

	# Grab azimuths and calculate a range based on number of gates
	az = np.array(datadict['start_az'] + [datadict['end_az'][-1]])
	rng = np.linspace(0, f.max_range, data.shape[-1] + 1)

	# Convert az,range to x,y
	xlocs = rng[:-1] * np.sin(np.deg2rad(az[1:, None]))/111.0 + 2.131
	ylocs = rng[:-1] * np.cos(np.deg2rad(az[1:, None]))/111.0 + 0.456

	# --- Calculate coordinates bounding init point based on surface vector(s) (radians) ---
	baseCrds = np.array([(1.0,0.5,0.0,1.0),(1.0,-0.5,0.0,1.0),(-0.125,-0.5,0.0,1.0),(-0.125,0.5,0.0,1.0),(1.0,0.5,0.0,1.0)]) 	#crds of bounding box (Gridded degrees)
	testLocBearing = 0
	testLoc = np.array([0,0])
	#comp_matrix(scale, rotation, shear, translation)
	TM = comp_matrix(np.ones(3), np.array([0,0, testLocBearing]), np.ones(3), np.pad(testLoc, (0, 1), 'constant'))
	polyVerts = TM.dot(baseCrds.T).T[:,:2]										#apply transformation Matrix, remove padding, and re-transpose
	print("PolyVerts (Lon_Lat): ", polyVerts)

	# --- Generate ROI from coordiantes (above) create 2D boolean array to mask with ---
	xp,yp = xlocs.flatten(),ylocs.flatten()
	points = np.vstack((xp,yp)).T
	path = Path(polyVerts)
	grid = path.contains_points(points)
	grid = grid.reshape(np.shape(xlocs))
	rDataMasked = np.ma.masked_array(data, np.invert(grid))

	# --- Clip our masked array, create sub-array of data and rotate ---
	i, j = np.where(grid)
	indices = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')
	rDataMaskedClip = data[indices]
	rDataMaskClip = grid[indices]
	print("dimsPreTM", rDataMaskedClip.shape)
	print("maskDimsPreTM", rDataMaskClip.shape)
	rDataMaskedClip = rDataMaskedClip*rDataMaskClip
	print("dimsPostTM", rDataMaskedClip.shape)

# Plot the data
	deltaX = 0#-236.541
	deltaY = 0#-50.616
	negXLim = -1#(-30 + deltaX )
	posXLim = 1#(111 + deltaX )
	negYLim = -1#(-111 + deltaY )
	posYLim = 1#(111 + deltaY )
	norm, cmap = colortables.get_with_steps(ctable, 16, 16)

	ax.pcolormesh(xlocs[indices], ylocs[indices], rDataMaskedClip, norm=norm, cmap=cmap)
	ax.set_aspect('equal', 'datalim')
	#ax.set_xlim(negXLim, posXLim)
	#ax.set_ylim(negYLim, posYLim)
	add_timestamp(ax, f.metadata['prod_time'], y=0.02, high_contrast=True)

#plt.show()
plt.savefig(args["output"] +'_Test_NRL3_.png') # Set the output file name

# --- Read in 2 datasets: 1) GFS U (Easting) component of wind velocity 2) GFS V (Northing) component of wind velocity ---										# array containing gridded LIS values

GFSGrbs = pygrib.open(args["GFS"])
UGrb = GFSGrbs.select(name='10 metre U wind component')[0]
print("\nU Data: " + str(UGrb))
U = UGrb.values * 1.944														# array containing U component gridded values (knots)
VGrb = GFSGrbs.select(name='10 metre V wind component')[0]
print("\nV Data: " + str(VGrb))
V = VGrb.values * 1.944														# array containing V component gridded values (knots)

ULons = np.linspace(float(UGrb['longitudeOfFirstGridPointInDegrees']), float(UGrb['longitudeOfLastGridPointInDegrees']), int(UGrb['Ni']) )
ULats = np.linspace(float(UGrb['latitudeOfFirstGridPointInDegrees']), float(UGrb['latitudeOfLastGridPointInDegrees']), int(UGrb['Nj']) )	
U, ULons = shiftgrid(180., U, ULons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
UGridLon, UGridLat = np.meshgrid(ULons, ULats) 								# regularly spaced 2D grid of GFS U component values

VLons = np.linspace(float(VGrb['longitudeOfFirstGridPointInDegrees']), float(VGrb['longitudeOfLastGridPointInDegrees']), int(VGrb['Ni']) )	
VLats = np.linspace(float(VGrb['latitudeOfFirstGridPointInDegrees']), float(VGrb['latitudeOfLastGridPointInDegrees']), int(VGrb['Nj']) )	
V, VLons = shiftgrid(180., V, VLons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
VGridLon, VGridLat = np.meshgrid(VLons, VLats) 								# regularly spaced 2D grid of GFS V component values

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

print("input Loc: ", testLoc)
print("Nearest GFS Vector Loc: ", gridTestLoc)
print("Nearest Vector --> Bearing: %3.2f from East (CCW) @ %3.1f knots (3x3 kernal gausian avg (sigma=1.0))" % (testLocBearing* 180 / np.pi, testLocMag))
print("Variance of wind components --> U: %3.2f  V: %3.2f" % (np.var(U),np.var(V)))
print("Variance of wind direction --> %3.2f " % (np.var(np.arctan2(testValV,testValU)*180/np.pi)))
