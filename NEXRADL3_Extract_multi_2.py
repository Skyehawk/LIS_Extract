# NEXRAD Data Extract - v0.11.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-03-27
#
# Updated
# 2020-04-28
#
# Developed as a tool to extract values from NEXRAD inputs for Thesis work
# 
# Use:	Linux call:  python /mnt/d/Libraries/Documents/Scripts/LIS_Plot/NEXRADL3_Extract_multi_2.py -r /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/NEXRAD/2016/20160715_2/L2 -g /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/GFS/2016/gfsanl_4_20160715_0600_000.grb2 -o /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/Output/2016/20160715_2 -c 41.55 -102.18 -s 41.95778 -100.57583


# --- Imports ---
import os
from os.path import join
import argparse
import numpy as np
import multiprocessing as mp
from datetime import datetime

from scipy.fft import fft
from scipy.signal import blackman

import matplotlib.pyplot as plt						# v. 3.1.0
import matplotlib.colors as colors
from matplotlib.path import Path
from matplotlib import dates as mpl_dates
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import shiftgrid

import pandas as pd
from metpy.plots import add_timestamp, colortables
import glob											# v0.7	
from tqdm import tqdm

from RadarSlice import RadarSlice
from RadarROI import RadarROI

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
ap.add_argument("-b", "--convBearing", nargs="+", help="passed: -b Bearing of storm training, in Rads, measured CCW from East")
ap.add_argument("-s", "--sensorLatLon", nargs="+", help="passed: -s <lat_float> <lon_float>; Lat/Lon of radar")
args = vars(ap.parse_args())

# --- Utility function(s) ---
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def calculate_radar_stats(d, filepath):
	roi = None
	if args["sensorLatLon"] is not None:
		roi = RadarROI(file=filepath, sensorData=np.array(args["sensorLatLon"]))
	else:
		roi = RadarROI(file=filepath, sensorData=None)

	offset = np.array([roi.sensorData["lat"] - float(args["convLatLon"][0]),
						roi.sensorData["lon"] - float(args["convLatLon"][1])])
	baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),(-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),(0.8750,0.25,0.0,1.0)]) 	#crds of bounding box (Gridded degrees)
	#testLocBearing = -.425

	roi.calc_cartesian()
	roi.shift_cart_orgin(offset=offset)

	#roi.extractROI(baseBearing=float(args["convBearing"][0]))			# General locating
	roi.extractROI(baseCrds=baseCrds, baseBearing=float(args["convBearing"][0]))
	
	reflectThresh = 139.0												# return strength threshold (139.0 = 35dbz)		
	roi.find_area(reflectThresh)
	roi.find_mean_reflectivity(reflectThresh)
	roi.find_variance_reflectivity(reflectThresh)
	d[roi.datetime] = [roi.datetime,roi.metadata,roi.sensorData,roi.mask,roi.xlocs,roi.ylocs,roi.clippedData,roi.polyVerts,offset,roi.area,roi.meanReflectivity,roi.varReflectivity]

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

	print('Sorting...')
	columns =['datetime', 'metadata', 'sensorData', 'indices', 'xlocs', 'ylocs', 'data', 'polyVerts', 'offset', 'areaValue', 'refValue', 'varRefValue']
	resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)
	resultsDF['datetime'] = pd.to_datetime(resultsDF.datetime)
	resultsDF.sort_values(by='datetime', inplace=True)
	resultsDF.to_csv(args["output"] + '.csv', index = False)
	print(resultsDF[['areaValue','refValue']].head(5))

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
		tempdata = record['data']									# create a deep copy of data to maipulate for plotting
		tempdata[tempdata == 0] = np.ma.masked						# mask out 0s for plotting
 
		axes[ploty][plotx].pcolormesh(record['xlocs'],record['ylocs'], tempdata, norm=norm, cmap=cmap)
		axes[ploty][plotx].set_aspect('equal', 'datalim')
		axes[ploty][plotx].set_xlim(negXLim, posXLim)
		axes[ploty][plotx].set_ylim(negYLim, posYLim)
		pVXs, pVYs = zip(*record['polyVerts'])						# create lists of x and y values for transformed polyVerts
		axes[ploty][plotx].plot(pVXs,pVYs)
		axes[ploty][plotx].plot(record['offset'][1], record['offset'][0], 'o')			# Location of the radar
		#axes[ploty][plotx].text(record['offset'][1], record['offset'][0], record['sensorData']['siteID'])		# will plot outside limits of subplot if site falls outside range
		axes[ploty][plotx].plot(0.0, 0.0, 'bx')						# Location of the convection
		axes[ploty][plotx].text(0.0, 0.0, str(args["convLatLon"]))
		add_timestamp(axes[ploty][plotx], record['datetime'], y=0.02, high_contrast=True)
		axes[ploty][plotx].tick_params(axis='both', which='both')

	print('Plotting Additional Data and Saving Output...')

	# pull data out of DF to make code cleaner
	datetimes = resultsDF['datetime'].tolist()
	elapsedtimes = list(map(lambda x: x - min(datetimes), datetimes))		# not currently used, need to get this wprking
	areaValues = resultsDF['areaValue'].tolist()					# area ≥ 35dbz within ROI
	refValues = resultsDF['refValue'].tolist()						# mean reflectivity ≥ 35dbz within ROI
	varValues = resultsDF['varRefValue'].tolist()					# variance of mean reflectivity ≥ 35dbz within ROI
	cvValues = [a / b for a, b in zip(varValues, refValues)]		# coeff. of variation for mean reflectivity ≥ 35dbz within ROI

	# frequency
	N = len(refValues)
	T = 1.0/N
	yf = fft(refValues)
	w = blackman(N)
	ywf = fft(refValues*w)
	xf = np.linspace(0,1.0/(2.0*T),N//2)

	window = 3														# number of samples in moving average
	yArea_avg = movingaverage(areaValues, window)					# create moving averages for time series'
	yRef_avg = movingaverage(refValues, window)
	yCV_avg = movingaverage(cvValues, window)

	# Area for Reflectivity ≥ 35dbz
	axes[-1][-4].plot_date(datetimes,areaValues,linestyle='solid', ms=4)
	axes[-1][-4].plot_date(datetimes[window:-window], yArea_avg[window:-window],"r", linestyle='solid')
	axes[-1][-4].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-4].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-4].set_title('Area of Reflectivity ≥ 35dbz')

	# TODO: map y axis to dbz for output
	# Mean of Reflectivity ≥ 35dbz
	axes[-1][-3].plot_date(datetimes,refValues,linestyle='solid', ms=4)
	axes[-1][-3].plot_date(datetimes[window:-window], yRef_avg[window:-window],"r", linestyle='solid')
	axes[-1][-3].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-3].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-3].set_title('Mean of Reflectivity ≥ 35dbz')
	
	# Coeff. of Variance of Reflectivity ≥ 35dbz
	axes[-1][-2].plot_date(datetimes,cvValues,linestyle='solid', ms=4)
	axes[-1][-2].plot_date(datetimes[window:-window], yCV_avg[window:-window],"r", linestyle='solid')
	axes[-1][-2].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-2].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-2].set_title('CV of Reflectivity ≥ 35dbz')

	# Testing plot
	axes[-1][-1].semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
	axes[-1][-1].semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
	axes[-1][-1].legend(['FFT','FFT w. Window'])
	#axes[-1][-1].plot(xf, 2.0/N * np.abs(yf[0:N//2]),linestyle='solid', ms=4)
	#axes[-1][-1].plot_date(datetimes[window:-window], yCV_avg[window:-window],"r", linestyle='solid')
	#axes[-1][-1].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-1].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-1].set_title('Testing Plot')

	plt.tight_layout()
	plt.savefig(args["output"] +'Nexrad.png') 						# Set the output file name
	#plt.show()

if __name__ == '__main__':
	main()