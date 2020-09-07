# NEXRAD Data Extract - v0.1.0
# Python Version: 3.7.3
#
# @author: Skye Leake
# @date: 2020-07-24
#
# Updated
# 2020-07-30
#
# Developed as a tool to extract values from NEXRAD L2 inputs for Thesis work
# 
# Use:	Example Linux call:  python NEXRADL3_Extract_multi.py -r /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/NEXRAD/2011/20110814_0/L2 -g /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/GFS/2011/gfsanl_4_20110815_0000_000.grb2 -o /mnt/d/Libraries/Documents/Grad_School/Thesis_Data/Output/2011/20110814_0 -c 45.68 -101.47 -b 0.2
#	Currently not working for 2015 and prior

# --- Imports ---
import os
from os.path import join
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as TPool
import datetime
from io import BytesIO
import gzip


import pandas as pd
from tqdm import tqdm
from scipy.fft import fft
from scipy.signal import blackman, argrelmax, argrelmin 
import matplotlib.pyplot as plt						# v. 3.1.0
import matplotlib.colors as colors
from matplotlib.path import Path
from matplotlib import dates as mpl_dates
from mpl_toolkits.basemap import Basemap, shiftgrid

from metpy.io import Level2File
from metpy.plots import add_timestamp, colortables, ctables

import boto3
import botocore
from botocore.client import Config

from RadarSlice_L2 import RadarSlice_L2
from RadarROI_L2 import RadarROI_L2

import logging
import warnings		# Debug (progress bars bugged by matplotlib futurewarnings output being annoying)
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Construct argument parse to parse the arguments (input dates) ---
ap = argparse.ArgumentParser()
#ap.add_argument("-r", "--NEXRADL2", help=" path to the input directory of NEXRADL2 Data if local()")
#ap.add_argument("-g", "--GFS", help=" path to the input file of GFS Wind U*V data (GRIB)")
#ap.add_argument("-e", "--extension", type=str, default="", help="(Optional) file extension. Default None")
ap.add_argument("-o", "--output", help=" path to the output directory")
ap.add_argument("-t", "--convTime", type=str, help="passed: -t <YYYYMMDDHHMMSS> Start time of observation (UTC)")
ap.add_argument("-i", "--convInterval", type=str, default='0200', help="passed: -i <HHMM> Period of observation measured from start time (inclusive)")
ap.add_argument("-d", "--convThreshMin", type=float, default='35.0', help="passed: -d Minimum threshold of reflectivity (dBZ) ")
ap.add_argument("-c", "--convLatLon", nargs="+", help="passed: -c <lat_float> <lon_float>; Lat/Lon of point of convection")
ap.add_argument("-b", "--convBearing", type=float, help="passed: -b Bearing of storm training, in Rads, measured CCW from East")
ap.add_argument("-s", "--sensor", help=" 4 letter code for sensor")
ap.add_argument("-sf", "--scaleFactor", type=float, default=1.0, help=" (Optional) Scale factor for ROI when performing sensitivity analysis")

ap.add_argument("-l", "--log", dest="logLevel", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="(default: %(default)s) Set the logging level")

args = ap.parse_args()
if args.logLevel:
	logging.basicConfig(level=getattr(logging, args.logLevel))

# --- Utility function(s) ---
def movingaverage(interval, window_size):
	'''
	Simple convolve smoothing for curves
	'''
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, 'same')

def pull_data(startDateTime, station):
	'''
	Pulls all radar data streams for a specified station (One hr incraments)
	Param:	<Datetime> from which to start 
			<String> station code
	Return:	<List> of s3 bucket handles which contains L2 radar data
			<List> of Datetimes which correspond to each bucket
	'''
	dt = startDateTime
	s3 = boto3.resource('s3', config=Config(signature_version=botocore.UNSIGNED, 
		user_agent_extra='Resource'))

	bucket = s3.Bucket('noaa-nexrad-level2')

	prefix = f'{dt:%Y}/{dt:%m}/{dt:%d}/{station}/{station}{dt:%Y%m%d_%H}'

	objects = []
	sweepDateTimes = []
	for obj in bucket.objects.filter(Prefix=prefix):
		objects.append(obj)
		sweepDateTimes.append(datetime.datetime.strptime(obj.key[20:35], '%Y%m%d_%H%M%S'))
	return objects, sweepDateTimes

def calculate_radar_stats(d, radarFile):
	'''
	Driver for the conversion and calculation of any stats in the radar objects, run in parellel with multiprocessing
	Param:  d <dict> Output for multiprocessing
			radarfile <metpy.io Level2File> 
	Return: None 
	'''
	roi = RadarROI_L2(radarFile=radarFile)

	sensors = {'KMVX':(47.52806, -97.325), 'KBIS':(46.7825, -100.7572), 
			'KMBX':(48.3925, -100.86444), 'KABR':(45.4433, -98.4134), 
			'KFSD':(43.5778, -96.7539), 'KUDX':(44.125, -102.82944), 
			'KOAX':(41.32028, -96.36639), 'KLNX':(41.95778, -100.57583), 
			'KUEX':(40.32083, -98.44167), 'KGLD':(39.36722, -101.69333),
			'KCYS':(41.15166, -104.80622)}

	offset = np.array([sensors[args.sensor][0] - float(args.convLatLon[0]),
						sensors[args.sensor][1] - float(args.convLatLon[1])])

	baseCrds = np.array([(0.8750,0.25,0.0,1.0),(0.8750,-0.25,0.0,1.0),
						(-0.125,-0.125,0.0,1.0),(-0.125,0.125,0.0,1.0),
						(0.8750,0.25,0.0,1.0)]) 	#crds of bounding box (Gridded degrees)

	roi.calc_cartesian()
	roi.shift_cart_orgin(offset=offset)

	#roi.extractROI(baseBearing=args.convBearing)			# General locating
	roi.extractROI(baseCrds=baseCrds, baseBearing=args.convBearing, scaleFactor=args.scaleFactor)
	
	reflectThresh = args.convThreshMin												# return strength threshold (135.0 = 35dbz)		
	roi.find_area(reflectThresh)
	roi.find_mean_reflectivity(reflectThresh)
	roi.find_variance_reflectivity(reflectThresh)
	d[roi.sweepDateTime] = [roi.sweepDateTime,roi.metadata,roi.sensorData,\
							roi.area,roi.meanReflectivity, roi.varReflectivity]

def main():
	manager = mp.Manager()
	results = manager.dict()
	pool = TPool(12)
	jobs = []

	startDateTime = datetime.datetime.strptime(args.convTime, '%Y%m%d%H%M')
	intervalDateTime = datetime.timedelta(hours=2, minutes=0)#hours = int(args.convInterval[:2]), minutes=int([args.convInterval[2:]]))

	station = args.sensor
	
	# Query all L2 files for the sensor
	totalRadarObjects = []
	totalSweepDateTimes = []
	hrIter = datetime.timedelta(hours=0)
	while True:																					# grab a specific interval of files
		radarObjects, sweepDateTimes = pull_data(startDateTime=(startDateTime+hrIter),\
												 station=station)
		totalRadarObjects.extend(radarObjects[:-1])
		totalSweepDateTimes.extend(sweepDateTimes[:-1])										# remove trailing *_MDM file
		if totalSweepDateTimes[-1] - startDateTime >= intervalDateTime:
			break
		else: 
			hrIter += datetime.timedelta(hours=1)
	fileDict = {'L2File':totalRadarObjects, 'Time':totalSweepDateTimes}
	fileDF = pd.DataFrame(fileDict)	
	print(f'Start time: {startDateTime}, Interval: {intervalDateTime}, End Time: {startDateTime + intervalDateTime}')

	filesToStream = fileDF[((fileDF['Time'] >= startDateTime) \
					& (fileDF['Time'] <= startDateTime + \
					intervalDateTime))]['L2File'].tolist()							# Bitwise operators, conditions double wrapped in perentheses to handle overriding
	print(f'files: {[obj.key for obj in filesToStream]}')
	if len(filesToStream) < 8:
		warnings.warn("n of radar inputs is not sufficent for curve smoothing",  UserWarning)

	# --- Stream files ahead of time to avoid error with multiprocessing and file handles ---
	filesToWorkers = []
	for L2FileStream in tqdm(filesToStream,desc="Streaming L2 Files"):
		if datetime.datetime.strptime(L2FileStream.key[20:35], '%Y%m%d_%H%M%S') >= datetime.datetime(2016, 1, 1):
			filesToWorkers.append(Level2File(L2FileStream.get()['Body']))
		else:
			bytestream = BytesIO(L2FileStream.get()['Body'].read())
			with gzip.open(bytestream, 'rb') as f:
				filesToWorkers.append(Level2File(f))  
				#filesToWorkers.append(Level2File(GzipFile(bytestream).read()))

	# --- Create pool for workers ---
	for file in filesToWorkers:
		job = pool.apply_async(calculate_radar_stats, (results, file))
		jobs.append(job)

	# --- Commit pool to workers ---
	for job in tqdm(jobs,desc="Bounding & Searching Data"):
		job.get()

	pool.close()
	pool.join()

	columns =['sweepDateTime', 'metadata', 'sensorData',
				'areaValue', 'refValue', 'varRefValue']
	print('Creating Dataframe... (This may take a while if plotting significant data)')
	resultsDF = pd.DataFrame.from_dict(results, orient='index', columns=columns)	#SUPER slow
	print('Converting datetimes...')
	resultsDF['sweepDateTime'] = pd.to_datetime(resultsDF.sweepDateTime)
	print('Sorting...')
	resultsDF.sort_values(by='sweepDateTime', inplace=True)
	#resultsDF.to_csv(args.output + '.csv', index = False)
	print(resultsDF[['areaValue','refValue']].head(5))

	# --- Plot time series---
	'''
	fig, axes = plt.subplots(8, 8, figsize=(30, 30))
	date_format = mpl_dates.DateFormatter('%H:%Mz')

	for i, (dt, record) in tqdm(enumerate(resultsDF.iterrows()), desc='Plotting Slices'):
		plotx = i%8
		ploty = int(i/8)

		negXLim = -.5
		posXLim = 1.5
		negYLim = -1.0
		posYLim = 1.0
		norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', 5,5)
		tempdata = record['data']									# create a deep copy of data to maipulate for plotting
		tempdata[tempdata == 0] = np.ma.masked						# mask out 0s for plotting
 
		axes[ploty][plotx].pcolormesh(record['xlocs'],record['ylocs'], tempdata, norm=norm, cmap=cmap, shading='auto')
		axes[ploty][plotx].set_aspect(aspect='equal')
		axes[ploty][plotx].set_xlim(negXLim, posXLim)
		axes[ploty][plotx].set_ylim(negYLim, posYLim)
		pVXs, pVYs = zip(*record['polyVerts'])						# create lists of x and y values for transformed polyVerts
		axes[ploty][plotx].plot(pVXs,pVYs)
		if negXLim < record['offset'][1] < posXLim and negYLim < record['offset'][0] < posYLim: 
			axes[ploty][plotx].plot(record['offset'][1], record['offset'][0], 'o')			# Location of the radar
			axes[ploty][plotx].text(record['offset'][1], record['offset'][0], record['sensorData']['siteID'])
			
		axes[ploty][plotx].plot(0.0, 0.0, 'bx')						# Location of the convection
		axes[ploty][plotx].text(0.0, 0.0, str(args.convLatLon))
		add_timestamp(axes[ploty][plotx], record['sweepDateTime'], y=0.02, high_contrast=True)
		axes[ploty][plotx].tick_params(axis='both', which='both')
	'''
	print('Calculating Statistics...')

	# pull data out of DF to make code cleaner
	datetimes = resultsDF['sweepDateTime'].tolist()
	#elapsedtimes = list(map(lambda x: x - min(datetimes), datetimes))						# not currently used, need to get this working
	areaValues = resultsDF['areaValue'].tolist()											# area ≥ 35dbz within ROI
	refValues = np.array(resultsDF['refValue'].tolist())									# mean reflectivity ≥ 35dbz within ROI (conversion: (val-65)*0.5) [https://mesonet.agron.iastate.edu/GIS/rasters.php?rid=2]
	if np.nan in refValues:
		 warnings.warn("Radar inputs contains instance with no ref values >= thresh",  UserWarning)
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
	print (f'Slope of Area: {slopeArea}')

	#   Reflectivity
	xRef = np.array(datetimes[window//2:-window//2])[np.array([refExtrema[0],refExtrema[1]])]
	xRefDiff = xRef[1] - xRef[0]
	yRef = yRefAvg[np.array([refExtrema[0], refExtrema[1]])]
	yRefDiff = yRef[1] - yRef[0]
	slopeRef = np.arctan(yRefDiff/xRefDiff.seconds)
	print (f'Slope of Reflectivity: {slopeRef}')

	# 	Product of Area and Coefficent of Variation of Reflectivity
	xProduct = np.array(datetimes[window//2:-window//2])[np.array([yAreaCVNormExtrema[0],yAreaCVNormExtrema[1]])]
	XProductDiff = xProduct[1] - xProduct[0]
	yProduct = yAreaCVNormAvg[np.array([yAreaCVNormExtrema[0],yAreaCVNormExtrema[1]])]
	yProductDiff = yProduct[1] - yProduct[0]
	slopeProduct = np.arctan(yProductDiff/XProductDiff.seconds)
	print (f'Slope of Product: {slopeProduct}')

	'''
	print('Plotting Additional Data and Saving Output...')
	# Area for Reflectivity ≥ 35dbz
	axes[-1][-5].plot_date(datetimes,areaValues,linestyle='solid', ms=2)
	axes[-1][-5].plot_date(datetimes[window//2:-window//2], yAreaAvg, linestyle='solid', ms=2)
	axes[-1][-5].plot_date(np.array(datetimes[window//2:-window//2])[np.array([areaExtrema[0],areaExtrema[1]])], yAreaAvg[np.array([areaExtrema[0],areaExtrema[1]])], linestyle="solid", ms=2)
	axes[-1][-5].legend(['Area Delta','Sm. Area Delta', 'Build-up Rate'])
	axes[-1][-5].xaxis.set_major_formatter(date_format)
	plt.setp(axes[-1][-5].xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor" )
	axes[-1][-5].set_title('Area of Reflectivity ≥ 35dbz (km^2)')

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
	axes[-1][-3].set_title('Norm Product:\nCV Reflectivity * Area ≥ 35dbz')

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
	plt.savefig(args.output +'Nexrad.png') 						# Set the output file name
	#plt.show()
	'''
	f_o = open(args.output + 'log_stats_area_nexrad.txt', 'a')
	f_o.write(datetimes[0].strftime("%Y%m%d%H%M%S")
		+ '\t' + str(args.convLatLon) + '\t' + str(args.convBearing) + '\t' + str(args.scaleFactor)
		+ '\t' + str(np.max(areaValues))
		+ '\t' + str(np.max(refValues))
		+ '\t' + str(slopeArea) 																				# std dev of LIS aligned data
		+ '\t' + str(slopeRef)
		+ '\t' + str(slopeProduct) + '\n')
	f_o.close()

if __name__ == '__main__':
	main()