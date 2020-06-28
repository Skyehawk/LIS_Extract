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
# Use:	Example Linux call:  python NEXRADL3_Extract_multi_2.py -r /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/NEXRAD/2011/20110814_0/L2 -g /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/GFS/2011/gfsanl_4_20110815_0000_000.grb2 -o /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/Output/2011/20110814_0 -c 45.68 -101.47 -b 0.2


# --- Imports ---
import os
from os.path import join
import argparse
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta

from scipy.fft import fft
from scipy.signal import blackman, argrelmax, argrelmin 

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
ap.add_argument("-b", "--convBearing", type=float, help="passed: -b bearing of storm training, in Rads, measured CCW from East")
ap.add_argument("-s", "--sensorLatLon", nargs="+", help="passed: -s <lat_float> <lon_float>; (Optional) Lat/Lon of radar if not in metadata")
ap.add_argument("-sf", "--scaleFactor", type=float, default=1.0, help=" (Optinal) Scale factor for ROI when performing sensitivity analysis")
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

	#roi.extractROI(baseBearing=args["convBearing"])			# General locating
	roi.extractROI(baseCrds=baseCrds, baseBearing=args["convBearing"], scaleFactor=args["scaleFactor"])
	
	reflectThresh = 135.0												# return strength threshold (135.0 = 35dbz)		
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
	#resultsDF.to_csv(args["output"] + '.csv', index = False)
	print(resultsDF[['areaValue','refValue']].head(5))

	# --- Plot time series---
	print('Plotting Slices...')
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
		if negXLim < record['offset'][1] < posXLim and negYLim < record['offset'][0] < posYLim: 
			axes[ploty][plotx].plot(record['offset'][1], record['offset'][0], 'o')			# Location of the radar
			axes[ploty][plotx].text(record['offset'][1], record['offset'][0], record['sensorData']['siteID'])		# will plot outside limits of subplot if site falls outside range
			
		axes[ploty][plotx].plot(0.0, 0.0, 'bx')						# Location of the convection
		axes[ploty][plotx].text(0.0, 0.0, str(args["convLatLon"]))
		add_timestamp(axes[ploty][plotx], record['datetime'], y=0.02, high_contrast=True)
		axes[ploty][plotx].tick_params(axis='both', which='both')

	print('Calculating Statistics...')

	# pull data out of DF to make code cleaner
	datetimes = resultsDF['datetime'].tolist()
	#elapsedtimes = list(map(lambda x: x - min(datetimes), datetimes))						# not currently used, need to get this working
	areaValues = resultsDF['areaValue'].tolist()											# area ≥ 35dbz within ROI
	refValues = (np.array(resultsDF['refValue'].tolist())-65) * 0.5							# mean reflectivity ≥ 35dbz within ROI (conversion: (val-65)*0.5) [https://mesonet.agron.iastate.edu/GIS/rasters.php?rid=2]
	#areaRefValues = np.multiply(areaValues, refValues)										# product of area and reflectivity
	varValues = resultsDF['varRefValue'].tolist()											# variance of mean reflectivity ≥ 35dbz within ROI
	cvValues = np.array([a / b for a, b in zip(varValues, refValues)])*0.5					# coeff. of variation for mean reflectivity ≥ 35dbz within ROI

	# Frequency
	N = len(refValues)
	T = 1.0/N
	yf = fft(refValues)
	w = blackman(N)
	ywf = fft(refValues*w)

	# Normalization
	areaNorm = areaValues / np.max(areaValues)
	xf = np.linspace(0,1.0/(2.0*T),N//2)
	cvNorm = cvValues / np.max(cvValues)
	areaCVValuesNormalized = np.multiply(areaNorm, cvNorm)

	# Curve Smoothing
	window = len(resultsDF.index)//8 														# ~2 hours/8 = ~15 mins ----> number of samples in moving average ( helps counteract more visible noise in higher temporal resolution data)
	yAreaAvg = movingaverage(areaValues, window)[window//2:-window//2]						# create moving averages for time series'
	yRefAvg = movingaverage(refValues, window)[window//2:-window//2]	
	yCVAvg = movingaverage(cvValues, window)[window//2:-window//2]	
	yAreaCVNormAvg = movingaverage(areaCVValuesNormalized, window)[window//2:-window//2]	

	# local minima & maxima on smoothed curves
	minTemporalwindow = window*2

	areaLocalMax = argrelmax(yAreaAvg)
	areaLocalMin = argrelmin(yAreaAvg)
	endpoints = []
	if yAreaAvg[0] <= np.all(yAreaAvg[1:window]) or yAreaAvg[0] >= np.all(yAreaAvg[1:window]):
		endpoints.append(0)
	if yAreaAvg[-1] <= np.all(yAreaAvg[len(yAreaAvg-1)-window:-2]) or yAreaAvg[-1] >= np.all(yAreaAvg[len(yAreaAvg-1)-window:-2]):
		endpoints.append(len(yAreaAvg)-1) 
	areaExtremaRaw = sorted(areaLocalMax[0].tolist()+areaLocalMin[0].tolist()+endpoints)	# combine mins, maxes, and endpoints (if endpoints are an extreme) then sort
	areaExtrema = [x for x in areaExtremaRaw[1:] if x-areaExtremaRaw[0]>=minTemporalwindow] # remove maxima that are within threshold of first one
	areaExtrema = [areaExtremaRaw[0]]+areaExtrema							# add back in forst one to begining
	print(f'Area Extrema: {areaExtrema}')

	refLocalMax = argrelmax(yRefAvg)
	refLocalMin = argrelmin(yRefAvg)
	endpoints = []
	if yRefAvg[0] <= np.all(yRefAvg[1:window]) or yRefAvg[0] >= np.all(yRefAvg[1:window]):
		endpoints.append(0)
	if yRefAvg[-1] <= np.all(yRefAvg[len(yRefAvg-1)-window:-2]) or yRefAvg[-1] >= np.all(yRefAvg[len(yRefAvg-1)-window:-2]):
		endpoints.append(len(yRefAvg)-1) 
	refExtremaRaw = sorted(refLocalMax[0].tolist()+refLocalMin[0].tolist()+endpoints)
	refExtrema = [x for x in refExtremaRaw[1:] if x-refExtremaRaw[0]>=minTemporalwindow]
	refExtrema = [refExtremaRaw[0]]+refExtrema
	print(f'Ref Extrema: {refExtrema}')
	
	#cvLocalMax = argrelmax(yCVAvg)
	#cvLocalMin = argrelmin(yCVAvg)
	#endpoints = []
	#if yCVAvg[0] <= np.all(yCVAvg[1:window]) or yCVAvg[0] >= np.all(yCVAvg[1:window]):
	#	endpoints.append(0)
	#if yCVAvg[-1] <= np.all(yCVAvg[len(yCVAvg-1)-window:-2]) or yCVAvg[-1] >= np.all(yCVAvg[len(yCVAvg-1)-window:-2]):
	#	endpoints.append(len(yCVAvg)-1) 
	#cvExtremaRaw = sorted(cvLocalMax[0].tolist()+cvLocalMin[0].tolist()+endpoints)
	#cvExtrema = [x for x in cvExtremaRaw[1:] if x-cvExtremaRaw[0]>=minTemporalwindow]
	#cvExtrema = [cvExtremaRaw[0]]+cvExtrema
	#print(f'CV Extrema: {cvExtrema}')

	yAreaCVNormLocalMax = argrelmax(yAreaCVNormAvg)
	yAreaCVNormLocalMin = argrelmin(yAreaCVNormAvg)
	endpoints = []
	if yAreaCVNormAvg[0] <= np.all(yAreaCVNormAvg[1:window]) or yAreaCVNormAvg[0] >= np.all(yAreaCVNormAvg[1:window]):
		endpoints.append(0)
	if yAreaCVNormAvg[-1] <= np.all(yAreaCVNormAvg[len(yAreaCVNormAvg-1)-window:-2]) or yAreaCVNormAvg[-1] >= np.all(yAreaCVNormAvg[len(yAreaCVNormAvg-1)-window:-2]):
		endpoints.append(len(yAreaCVNormAvg)-1) 
	yAreaCVNormExtremaRaw = sorted(yAreaCVNormLocalMax[0].tolist()+yAreaCVNormLocalMin[0].tolist()+endpoints)
	yAreaCVNormExtrema = [x for x in yAreaCVNormExtremaRaw[1:] if x-yAreaCVNormExtremaRaw[0]>=minTemporalwindow]
	yAreaCVNormExtrema = [yAreaCVNormExtremaRaw[0]]+yAreaCVNormExtrema
	print(f'AreaCVNorm Extrema: {yAreaCVNormExtrema}')
	

	# Find slopes of Build-up Lines
	# 	Area
	xArea = np.array(datetimes[window//2:-window//2])[np.array([areaExtrema[0],areaExtrema[1]])]		# grab datetime (x component) of the leftmost bounds (determined by window size), and the first extreme on the smoothed curve (sm curve is already bound by window, we need to apply bounds to datetimes)
	xAreaDiff = xArea[1] - xArea[0]															# subtract the later value from the former to get our delta x
	yArea = yAreaAvg[np.array([areaExtrema[0],areaExtrema[1]])]								# grab the values (y component) of the sm curve at the begining and at the first extreme
	yAreaDiff = yArea[1] - yArea[0]															# subtract to find delta y
	slopeArea = np.arctan(yAreaDiff/xAreaDiff.seconds)										# calc the slope angle
	print (f'Slope Area: {slopeArea}')

	#   Reflectivity
	xRef = np.array(datetimes[window//2:-window//2])[np.array([refExtrema[0],refExtrema[1]])]
	xRefDiff = xRef[1] - xRef[0]
	yRef = yRefAvg[np.array([refExtrema[0], refExtrema[1]])]
	yRefDiff = yRef[1] - yRef[0]
	slopeRef = np.arctan(yRefDiff/xRefDiff.seconds)
	print (f'Slope Reflectivity: {slopeRef}')

	# 	Product of Area and CV of ref
	xProduct = np.array(datetimes[window//2:-window//2])[np.array([yAreaCVNormExtrema[0],yAreaCVNormExtrema[1]])]
	XProductDiff = xProduct[1] - xProduct[0]
	yProduct = yAreaCVNormAvg[np.array([yAreaCVNormExtrema[0],yAreaCVNormExtrema[1]])]
	yProductDiff = yProduct[1] - yProduct[0]
	slopeProduct = np.arctan(yProductDiff/XProductDiff.seconds)
	print (f'Slope Product: {slopeProduct}')

	print('Plotting Additional Data and Saving Output...')
	# Area for Reflectivity ≥ 35dbz
	axes[-1][-5].plot_date(datetimes,areaValues,linestyle='solid', ms=2)
	axes[-1][-5].plot_date(datetimes[window//2:-window//2], yAreaAvg, linestyle='solid', ms=2)
	axes[-1][-5].plot_date(np.array(datetimes[window//2:-window//2])[np.array([areaExtrema[0],areaExtrema[1]])], yAreaAvg[np.array([areaExtrema[0],areaExtrema[1]])], linestyle="solid", ms=2)
	axes[-1][-5].legend(['Area Delta','Sm. Area Delta', 'Build-up Rate'])
	axes[-1][-5].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-5].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-5].set_title('Area of Reflectivity ≥ 35dbz (km^2)')

	# TODO: map y axis to dbz for output
	# Mean of Reflectivity ≥ 35dbz
	axes[-1][-4].plot_date(datetimes,refValues,linestyle='solid', ms=2)
	#axes[-1][-4].plot_date(datetimes[window//2:-window//2], yRefAvg, linestyle='solid', ms=2)
	#axes[-1][-4].plot_date(np.array(datetimes[window//2:-window//2])[np.array([0,refLocalMax[0][0]])], yRefAvg[np.array([0,refLocalMax[0][0]])], linestyle="solid", ms=2)
	axes[-1][-4].plot_date(datetimes[window//2:-window//2], yRefAvg, linestyle='solid', ms=2)
	axes[-1][-4].plot_date(np.array(datetimes[window//2:-window//2])[np.array([refExtrema[0],refExtrema[1]])], yRefAvg[np.array([refExtrema[0],refExtrema[1]])], linestyle="solid", ms=2)
	axes[-1][-4].legend(['Ref Delta','Sm. Ref Delta', 'Build-up Rate'])
	axes[-1][-4].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-4].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-4].set_title('Mean of Reflectivity ≥ 35dbz')
	
	# Product of cv reflectivity and area
	axes[-1][-3].plot_date(datetimes,areaCVValuesNormalized,linestyle='solid', ms=2)
	axes[-1][-3].plot_date(datetimes[window//2:-window//2], yAreaCVNormAvg, linestyle='solid', ms=2)
	axes[-1][-3].plot_date(np.array(datetimes[window//2:-window//2])[np.array([yAreaCVNormExtrema[0],yAreaCVNormExtrema[1]])], yAreaCVNormAvg[np.array([yAreaCVNormExtrema[0],yAreaCVNormExtrema[1]])], linestyle="solid", ms=2)
	axes[-1][-3].legend(['Area*cv_Ref Delta','Sm. Area*cv_Ref Delta', 'Build-up Rate'])
	axes[-1][-3].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-3].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-3].set_title('Norm Product: CV Reflectivity * Area ≥ 35dbz')

	# Coeff. of Variance of Reflectivity ≥ 35dbz
	axes[-1][-2].plot_date(datetimes,cvValues,linestyle='solid', ms=2)
	axes[-1][-2].plot_date(datetimes[window//2:-window//2], yCVAvg, linestyle='solid', ms=2)
	axes[-1][-2].legend(['CV Delta','Sm. CV Delta'])
	axes[-1][-2].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-2].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-2].set_title('CV of Reflectivity ≥ 35dbz')

	# Testing plot
	axes[-1][-1].semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
	axes[-1][-1].semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
	axes[-1][-1].legend(['FFT','FFT w. Window'])
	#axes[-1][-1].plot(xf, 2.0/N * np.abs(yf[0:N//2]),linestyle='solid', ms=2)
	#axes[-1][-1].plot_date(datetimes, yCVAvg, linestyle='solid')
	#axes[-1][-1].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-1].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-1].set_title('Testing Plot (Frequency)')

	plt.tight_layout()
	plt.savefig(args["output"] +'Nexrad.png') 						# Set the output file name
	#plt.show()

	f_o = open(args["output"] + 'log_stats_area_nexrad.txt', 'a')
	f_o.write(datetimes[0].strftime("%Y%m%d%H%M%S")
		+ '\t' + str(args["convLatLon"]) + '\t' + str(args["convBearing"]) + '\t' + str(args["scaleFactor"])
		+ '\t' + str(np.max(areaValues))
		+ '\t' + str(np.max(refValues))
		+ '\t' + str(slopeArea) 																				# std dev of LIS aligned data
		+ '\t' + str(slopeRef)
		+ '\t' + str(slopeProduct) + '\n')
	f_o.close()

if __name__ == '__main__':
	main()