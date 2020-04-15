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
# Use:	Linux call:  python /mnt/d/Libraries/Documents/Scripts/LIS_Plot/NEXRADL3_Extract.py -r /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/NEXRAD/2016/20160715_2/L2 -g /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/GFS/2016/gfsanl_4_20160715_0600_000.grb2 -o /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/Output/2016/20160715_2 -c 41.55 -102.13 -s 41.95778 -100.57583
#		
#
# Notes: Output currently .csv & ascii (in .txt format) files

# --- Imports ---
import os
from os.path import join
import argparse
import numpy as np
import multiprocessing as mp
import pygrib
from datetime import datetime
import matplotlib.pyplot as plt						# v. 3.1.0
import matplotlib.colors as colors
from matplotlib.path import Path
from matplotlib import dates as mpl_dates
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid
from skimage.transform import rotate				# v. 0.15.0
import pandas as pd
#from metpy.cbook import get_test_data
from metpy.io import Level3File
from metpy.plots import add_timestamp, colortables
import glob											# v0.7	
from tqdm import tqdm

from Transformation_Matrix_2 import comp_matrix
from Gauss_Map import gauss_map 

# Debug (progress bars bugged by matplotlib futurewarnings output being annoying)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Construct argument parse to parse the arguments (input dates) ---
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--NEXRADL3", help=" path to the input directory of NEXRADL3 Data ()")
ap.add_argument("-g", "--GFS", help=" path to the input file of GFS Wind U*V data (GRIB)")
ap.add_argument("-e", "--extension", type=str, default="", help="(Optional) file extension. Default None")
ap.add_argument("-o", "--output", help=" path to the output directory")
ap.add_argument("-c", "--convLatLon", nargs="+", help="passed: -c <lat_float> <lon_float>; Lat/Lon of point of convection")
ap.add_argument("-s", "--sensorLatLon", nargs="+", help="passed: -s <lat_float> <lon_float>; Lat/Lon of radar")
args = vars(ap.parse_args())

# --- Utility function(s) ---

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def writer(q):
	tempDF = pd.DataFrame(columns =['datetime', 'data', 'areaValue', 'refValue'])
	while 1:
		vals = q.get()
		if vals == 'shutdown':
			print('Shutdown message recieved')
			break
		row = pd.DataFrame(vals, columns =['datetime', 'data', 'areaValue', 'refValue'])
		tempDF = pd.concat([tempDF, row], ignore_index=True)
	return tempDF

# --- Get mean flow of cloud layer (Vcl) --- 

#def get_levels_vectors(u,v,loc=0,bounds=0):
#	UGrb = GFSGrbs.select()[u]
#	#print("\nU Data: " + str(UGrb))
#	U = UGrb.values * 1.944											# array containing U component gridded values (knots)
#	#VGrb = GFSGrbs.select(name=v)[0]
#	VGrb = GFSGrbs.select()[v]
#	#print("\nV Data: " + str(VGrb))
#	V = VGrb.values * 1.944											# array containing V component gridded values (knots)
#
#	ULons = np.linspace(float(UGrb['longitudeOfFirstGridPointInDegrees']), float(UGrb['longitudeOfLastGridPointInDegrees']), int(UGrb['Ni']) )
#	ULats = np.linspace(float(UGrb['latitudeOfFirstGridPointInDegrees']), float(UGrb['latitudeOfLastGridPointInDegrees']), int(UGrb['Nj']) )	
#	U, ULons = shiftgrid(180., U, ULons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
#	UGridLon, UGridLat = np.meshgrid(ULons, ULats) 								# regularly spaced 2D grid of GFS U component values
#
#	VLons = np.linspace(float(VGrb['longitudeOfFirstGridPointInDegrees']), float(VGrb['longitudeOfLastGridPointInDegrees']), int(VGrb['Ni']) )	
#	VLats = np.linspace(float(VGrb['latitudeOfFirstGridPointInDegrees']), float(VGrb['latitudeOfLastGridPointInDegrees']), int(VGrb['Nj']) )	
#	V, VLons = shiftgrid(180., V, VLons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
#	#VGridLon, VGridLat = np.meshgrid(VLons, VLats) 							#disabled - only used for plotting		# regularly spaced 2D grid of GFS V component values
#
#	testLoc = np.array([float(args["convLatLon"][1]),float(args["convLatLon"][0])])
#	gridTestLoc = np.around(testLoc*2)/2										# round to the nearest 1.0 for GFS3, 0.5 for GFS4
#	testLocIdx= np.where(gridTestLoc[1]==UGridLat)[0][0], np.where(gridTestLoc[0]==UGridLon)[1][0]
#	testValU = U[testLocIdx[0]-1: testLocIdx[0] + 2,testLocIdx[1]-1: testLocIdx[1] + 2]			# 3x3 sample of U values centered about our closest vector
#	testValV = V[testLocIdx[0]-1: testLocIdx[0] + 2,testLocIdx[1]-1: testLocIdx[1] + 2]			# 3x3 sample of V values centered about our closest vector
#	gausKern3x3sig1 = np.array([[0.077847,0.123317,0.077847],\
#							[0.123317,0.195346,0.123317],\
#							[0.077847,0.123317,0.077847]])
#	testLocBearing = np.arctan2(np.sum(testValV*gausKern3x3sig1), np.sum(testValU*gausKern3x3sig1))
#	testLocMag = np.sqrt(np.sum(testValV*gausKern3x3sig1)**2 + np.sum(testValU*gausKern3x3sig1)**2)
#
#	return np.array([testLocBearing*180/np.pi,testLocMag])


#GFSGrbs = pygrib.open(args["GFS"])
#compLayers = [(192,193),(176,177),(144,145),(109,110),(264,265)]

#layersVals = []
#for layer in tqdm(compLayers, desc='Finding Vcl: '):
#	layersVals.append(get_levels_vectors(layer[0],layer[1]))
#layerMeans = np.mean(np.array(layersVals[:4]),axis=0)
#print('Layer Values: ',layersVals)
#print('Layer Means: ',layerMeans)
#print('Layer Means adj: ',layerMeans - np.array([layersVals[4][0],0]))



def worker(filepath, q):

	offset = np.array([float(args["sensorLatLon"][0]) - float(args["convLatLon"][0]),
						float(args["sensorLatLon"][1]) - float(args["convLatLon"][1])])
	#print('offset (Lat, Lon):', offset)
	reflectThresh = 139												# return strength threshold (139.0 = 35dbz)


	#datetimes.append(datetime.strptime(filepath[-12:],'%Y%m%d%H%M'))
	f = Level3File(filepath)
	dataDict = f.sym_block[0][0]									# Pull the data out of the file object
	data = np.ma.array(dataDict['data'])							# Turn into an array, then mask
	#data[data == 0] = np.ma.masked 								# convert 0s to masked
	az = np.array(dataDict['start_az'] + [dataDict['end_az'][-1]])	# Grab azimuths and calculate a range based on number of gates
	rng = np.linspace(0, f.max_range, data.shape[-1] + 1)

	xlocs = (rng[:-1] * np.sin(np.deg2rad(az[1:, None]))/111.0) + offset[1]	# Convert az,range to x,y
	ylocs = (rng[:-1] * np.cos(np.deg2rad(az[1:, None]))/111.0) + offset[0]

	# --- Calculate coordinates bounding init point based on surface vector(s) (radians) ---
	baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),(-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),(0.8750,0.25,0.0,1.0)]) 	#crds of bounding box (Gridded degrees)
	#baseCrds = np.array([(5.0,5,0.0,1.0),(5.0,-0.5,0.0,1.0),(-5.0,-5.0,0.0,1.0),(-5.0,5.0,0.0,1.0),(5.0,5.0,0.0,1.0)])
	testLocBearing = -.5
	testLoc = np.array([0,0])										# Offsets have been delt with earlier by adding in the differance of radar loc and convection init loc, leave this as is
	TM = comp_matrix(np.ones(3), np.array([0,0, testLocBearing]), np.ones(3), np.pad(testLoc, (0, 1), 'constant'))
	polyVerts = TM.dot(baseCrds.T).T[:,:2]							# Apply transformation Matrix, remove padding, and re-transpose

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
	rDataMaskedClip = rDataMaskedClip*rDataMaskClip
	resArea = sum(map(lambda i: i >= reflectThresh, rDataMaskedClip.flatten()))
	resRef = np.mean(np.array(list(filter(lambda x: x >= reflectThresh, rDataMaskedClip.flatten()))))
	q.put([[f.metadata['prod_time'],rDataMaskedClip,resArea,resRef]])
	# no return()

	# --- Add to Plot (Debugging) ---
#	plotx = currfig%8
#	ploty = int(currfig/8)
#	currfig += 1

#	negXLim = -.5
#	posXLim = 1.5
#	negYLim = -1.0
#	posYLim = 1.0
#	norm, cmap = colortables.get_with_steps('NWSReflectivity', 18, 16)
#	tempdata = np.copy(rDataMaskedClip)								# create a deep copy of data to maipulate for plotting
#	#tempdata[tempdata == 0] = ma.masked
#	axes[ploty][plotx].pcolormesh(xlocs[indices], ylocs[indices], tempdata, norm=norm, cmap=cmap)
#	axes[ploty][plotx].set_aspect('equal', 'datalim')
#	axes[ploty][plotx].set_xlim(negXLim, posXLim)
#	axes[ploty][plotx].set_ylim(negYLim, posYLim)

#	pVXs, pVYs = zip(*polyVerts)									# create lists of x and y values for transformed polyVerts
#	axes[ploty][plotx].plot(pVXs,pVYs)
#	axes[ploty][plotx].plot(offset[1], offset[0], 'o')				# Location of the Radar
#	axes[ploty][plotx].plot(0.0, 0.0, 'bx')				# Location of the Convection
#	add_timestamp(axes[ploty][plotx], f.metadata['prod_time'], y=0.02, high_contrast=True)
#	axes[ploty][plotx].tick_params(axis='both', which='both')

# --- Global Vals ---
#results = []
#resultsDF = pd.DataFrame(columns =['datetime', 'data', 'areaValue', 'refValue'])

def main():
	manager = mp.Manager()
	q = manager.Queue()
	pool = mp.Pool(8)
	rawResultsDF = pool.apply_async(writer, (q,))
	jobs = []

	for filepath in glob.glob(join(args["NEXRADL3"],'*')):
		job = pool.apply_async(worker, (filepath,q))
		jobs.append(job)

	for job in tqdm(jobs,desc="Bounding & Searching Data"):
		job.get()

	q.put('shutdown')
	pool.close()
	pool.join()

	#print("entering sd")
	
	#print(f'results: {len(results)}')
	#resultsArray = np.array(results)
	#resArea = []
	#resRef = []
	#for arr in tqdm(results, desc="Searching Data"):
	#	print(f'arr: {arr}')
	#	resArea.append(sum(map(lambda i: i >= reflectThresh, arr[0].flatten())))
	#	resRef.append(np.mean(np.array(list(filter(lambda x: x >= reflectThresh, arr[1].flatten())))))
	#valsDF = pd.DataFrame(list(zip(arr[0],resArea,resRef)), columns =['datetime', 'areaValue', 'refValue']) # currently broken due to setting 0s to masked elements (line ~117)
	#valsDF.to_csv(args["output"] + '.csv', index = False)
	resultsDF = rawResultsDF.get().copy(deep=True)
	resultsDF['datetime'] = pd.to_datetime(resultsDF.datetime)
	resultsDF.sort_values(by='datetime', inplace=True)
	print(resultsDF)


#	window = 5
#	yArea_av = movingaverage(resArea, window)							# Create moving averages for time series'
#	yRef_av = movingaverage(resRef, window)

#	# --- Plot time series---
#	fig, axes = plt.subplots(8, 8, figsize=(30, 30))
#	date_format = mpl_dates.DateFormatter('%H:%Mz')
#
#	axes[-1][-2].plot_date(datetimes,resArea,linestyle='solid')
#	axes[-1][-2].plot_date(datetimes[window:-window], yArea_av[window:-window],"r", linestyle='solid')
#	axes[-1][-2].xaxis.set_major_formatter(date_format)
#	plt.setp(axes[-1][-2].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
#	axes[-1][-2].set_title('Area ≥ 35dbz')
#
#	axes[-1][-1].plot_date(datetimes,resRef,linestyle='solid')
#	axes[-1][-1].plot_date(datetimes[window:-window], yRef_av[window:-window],"r", linestyle='solid')
#	axes[-1][-1].xaxis.set_major_formatter(date_format)
#	plt.setp(axes[-1][-1].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
#	axes[-1][-1].set_title('Mean of Reflectivity ≥ 35dbz')
#	# TODO: convert y axis to dbz
#
#	plt.tight_layout()
#	plt.savefig(args["output"] +'Nexrad.png') # Set the output file name
#
if __name__ == '__main__':
	main()