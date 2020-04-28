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

from RadarSlice import RadarSlice
from RadarROI import RadarROI

# Debug (progress bars bugged by matplotlib futurewarnings output being annoying)
from MeasureDuration import MeasureDuration
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

def calculate_radar_stats(d, filepath):

	offset = np.array([float(args["sensorLatLon"][0]) - float(args["convLatLon"][0]),
						float(args["sensorLatLon"][1]) - float(args["convLatLon"][1])])
	baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),(-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),(0.8750,0.25,0.0,1.0)]) 	#crds of bounding box (Gridded degrees)
	testLocBearing = -.5

	roi = None		#debug
	roi = RadarROI(file=filepath,sensorLocation=np.array([0.0,0.0]))
	roi.calc_cartesian()
	roi.shift_cart_orgin(offset=offset)
	roi.extractROI(baseCrds=baseCrds, baseBearing=testLocBearing)
	reflectThresh = 139												# return strength threshold (139.0 = 35dbz)		
	roi.find_mean_reflectivity(reflectThresh)
	roi.find_area(reflectThresh)
	d[roi.datetime] = [roi.datetime,roi.mask,roi.xlocs,roi.ylocs,roi.clippedData,roi.polyVerts,offset,roi.area,roi.meanReflectivity]

def main():
	manager = mp.Manager()
	results = manager.dict()
	pool = mp.Pool(12)

	jobs = []

	for filepath in glob.glob(join(args["NEXRADL3"],'*')):
		job = pool.apply_async(calculate_radar_stats, (results, filepath))
		jobs.append(job)

	for job in tqdm(jobs,desc="Bounding & Searching Data"):
		job.get()

	pool.close()
	pool.join()

	columns =['datetime','indices', 'xlocs', 'ylocs', 'data', 'polyVerts', 'offset', 'areaValue', 'refValue']
	resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)
	resultsDF['datetime'] = pd.to_datetime(resultsDF.datetime)
	resultsDF.sort_values(by='datetime', inplace=True)
	#resultsDF.to_csv(args["output"] + '.csv', index = False)
	print(resultsDF[['areaValue','refValue']])

	# --- Plot time series---
	fig, axes = plt.subplots(8, 8, figsize=(30, 30))
	date_format = mpl_dates.DateFormatter('%H:%Mz')

	for i, (dt, record) in tqdm(enumerate(resultsDF.iterrows()), desc='Plotting Slices'):
		plotx = i%8
		ploty = int(i/8)

		negXLim = -.5
		posXLim = 1.5
		negYLim = -1.0
		posYLim = 1.0
		norm, cmap = colortables.get_with_steps('NWSReflectivity', 18, 16)
		tempdata = record['data']								# create a deep copy of data to maipulate for plotting
		#indices = record['data'][2]
		#loc = record['data'][0]
		#tempdata[tempdata == 0] = ma.masked

		# x and y refer to xlocs and ylocs ... we need to get these passed. 
		axes[ploty][plotx].pcolormesh(record['xlocs'],record['ylocs'], tempdata, norm=norm, cmap=cmap)
		axes[ploty][plotx].set_aspect('equal', 'datalim')
		axes[ploty][plotx].set_xlim(negXLim, posXLim)
		axes[ploty][plotx].set_ylim(negYLim, posYLim)
		pVXs, pVYs = zip(*record['polyVerts'])								# create lists of x and y values for transformed polyVerts
		axes[ploty][plotx].plot(pVXs,pVYs)
		axes[ploty][plotx].plot(record['offset'][1], record['offset'][0], 'o')			# Location of the radar
		axes[ploty][plotx].plot(0.0, 0.0, 'bx')						# Location of the convection
		add_timestamp(axes[ploty][plotx], record['datetime'], y=0.02, high_contrast=True)
		axes[ploty][plotx].tick_params(axis='both', which='both')

	window = 5
	yArea_av = movingaverage(resultsDF['areaValue'].to_list(), window)							# Create moving averages for time series'
	yRef_av = movingaverage(resultsDF['refValue'].to_list(), window)

	#TODO: Fix time series plots (commented out)
	#axes[-1][-2].plot_date(record['datetime'],record['areaValue'],linestyle='solid')
	#axes[-1][-2].plot_date(datetimes[window:-window], yArea_av[window:-window],"r", linestyle='solid')
	#axes[-1][-2].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-2].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-2].set_title('Area ≥ 35dbz')

	#axes[-1][-1].plot_date(datetimes,resRef,linestyle='solid')
	#axes[-1][-1].plot_date(datetimes[window:-window], yRef_av[window:-window],"r", linestyle='solid')
	#axes[-1][-1].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-1].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-1].set_title('Mean of Reflectivity ≥ 35dbz')
	# TODO: convert y axis to dbz

	plt.tight_layout()
	plt.savefig(args["output"] +'Nexrad.png') # Set the output file name

if __name__ == '__main__':
	main()