# LIS Data Extract - v0.10
# Python Version: 3.7.3
#
# Skye Leake
# 2019-11-27
#
# Updated
# 2020-03-24
#
# Developed as a tool to extract values from a grb file for overlay geostatistical analysis
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
fig, axes = plt.subplots(1, 1, figsize=(15, 8))
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
	print(np.shape(xlocs))
	print(np.shape(data))
	#print(rGriddedVals[:10,:10])
	





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
	#ax.pcolormesh(xlocs, ylocs, data, norm=norm, cmap=cmap)
	ax.set_aspect('equal', 'datalim')
	ax.set_xlim(negXLim, posXLim)
	ax.set_ylim(negYLim, posYLim)
	add_timestamp(ax, f.metadata['prod_time'], y=0.02, high_contrast=True)

#plt.show()
plt.savefig(args["output"] +'_Test_NRL3_.png') # Set the output file name






# --- Read in 3 datasets: 1) LIS Data for layer of intrest 2) GFS U (Easting) component of wind velocity 3) GFS V (Northing) component of wind velocity ---
LISGrbs = pygrib.open(args["LIS"])
LISGrb = LISGrbs.select()[32]												# index positions of relevent data
print("\nLIS Data: " + str(LISGrb))
LISData = LISGrb.values														# array containing gridded LIS values

GFSGrbs = pygrib.open(args["GFS"])
UGrb = GFSGrbs.select(name='10 metre U wind component')[0]
print("\nU Data: " + str(UGrb))
U = UGrb.values * 1.944														# array containing U component gridded values (knots)
VGrb = GFSGrbs.select(name='10 metre V wind component')[0]
print("\nV Data: " + str(VGrb))
V = VGrb.values * 1.944														# array containing V component gridded values (knots)

# --- Convert gridded data to lat/lon information ---
LISLons = np.linspace(float(LISGrb['longitudeOfFirstGridPointInDegrees']), float(LISGrb['longitudeOfLastGridPointInDegrees']), int(LISGrb['Ni']) )	#generate x coordinates normalized to the range of data
LISLats = np.linspace(float(LISGrb['latitudeOfFirstGridPointInDegrees']), float(LISGrb['latitudeOfLastGridPointInDegrees']), int(LISGrb['Nj']) )	#generate y coordinates normalized to the range of data
LISGridLon, LISGridLat = np.meshgrid(LISLons, LISLats) 						# regularly spaced 2D grid of LIS values

ULons = np.linspace(float(UGrb['longitudeOfFirstGridPointInDegrees']), float(UGrb['longitudeOfLastGridPointInDegrees']), int(UGrb['Ni']) )
ULats = np.linspace(float(UGrb['latitudeOfFirstGridPointInDegrees']), float(UGrb['latitudeOfLastGridPointInDegrees']), int(UGrb['Nj']) )	
U, ULons = shiftgrid(180., U, ULons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
UGridLon, UGridLat = np.meshgrid(ULons, ULats) 								# regularly spaced 2D grid of GFS U component values

VLons = np.linspace(float(VGrb['longitudeOfFirstGridPointInDegrees']), float(VGrb['longitudeOfLastGridPointInDegrees']), int(VGrb['Ni']) )	
VLats = np.linspace(float(VGrb['latitudeOfFirstGridPointInDegrees']), float(VGrb['latitudeOfLastGridPointInDegrees']), int(VGrb['Nj']) )	
V, VLons = shiftgrid(180., V, VLons, start=False)  							# use this to bump data from 0..360 on the lons to -180..180
VGridLon, VGridLat = np.meshgrid(VLons, VLats) 								# regularly spaced 2D grid of GFS V component values

# get wind vector from the nearest grid cell
#testLoc = np.array([-102.210999, 43.824593])								#2017/06/11
#testLoc = np.array([-99.184, 41.217])										#2017/06/14
#testLoc = np.array([-98.066, 45.456])										#2017/07/25

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

# --- Calculate coordinates bounding init point based on surface vector(s) (radians) ---
baseCrds = np.array([(1.0,1.0,0.0,1.0),(1.0,-1.0,0.0,1.0),(-1.0,-1.0,0.0,1.0),(-1.0,1.0,0.0,1.0),(1.0,1.0,0.0,1.0)]) 	#crds of bounding box (Gridded degrees)

#comp_matrix(scale, rotation, shear, translation)
TM = comp_matrix(np.ones(3), np.array([0,0, testLocBearing]), np.ones(3), np.pad(testLoc, (0, 1), 'constant'))
polyVerts = TM.dot(baseCrds.T).T[:,:2]										#apply transformation Matrix, remove padding, and re-transpose
print("PolyVerts (Lon_Lat): ", polyVerts)

# --- Generate ROI from coordiantes (above) create 2D boolean array to mask with ---
xp,yp = LISGridLon.flatten(),LISGridLat.flatten()
points = np.vstack((xp,yp)).T
path = Path(polyVerts)
grid = path.contains_points(points)
grid = grid.reshape(np.shape(LISGridLon))
LISDataMasked = np.ma.masked_array(LISData, np.invert(grid))

# --- Clip our masked array, create sub-array of data and rotate ---
i, j = np.where(grid)
indices = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')
LISDataMaskedClip = LISData[indices]
LISDataMaskClip = grid[indices]
#print("dimsPreRot", LISDataMaskedClip.shape)
#print("maskDimsPreRot", LISDataMaskClip.shape)
LISDataMaskedClip = LISDataMaskedClip*LISDataMaskClip
LISDataMaskedClip[LISDataMaskedClip > 100] = np.nan									# Replace our no data "9999" values with "nan"
LISAligned = rotate(LISDataMaskedClip, (testLocBearing * 180 / np.pi)-(90), resize=True, order=0)		# rotate about center with nearest neighbor parameters, offest 0deg to 90 (top-down wind vector)
LISAligned[LISAligned == 0.0] = np.nan							# Replace out of ROI data w/ "nan"
print('Alignment Checksum - Masked (unrotated) %9.0f Aligned (rotated) %9.0f' %(np.nansum(LISDataMaskedClip),np.nansum(LISAligned)))
LISAligned = LISAligned[int((LISAligned.shape[0]-68)/2):int(((LISAligned.shape[0]-68)/2)+68)\
						,int((LISAligned.shape[1]-68)/2):int(((LISAligned.shape[1]-68)/2)+68)]

# --- Create basic stats ---
LISAlignedMeanDep = LISAligned - np.nanmean(LISAligned)			# calculate our mean departures

gmap=gauss_map(size_x=np.shape(LISAligned)[0], size_y = np.shape(LISAligned)[1], sigma_x=10, sigma_y=20)
LISAlignedWeighted = abs(LISAlignedMeanDep * gmap)				# apply weaghted dist. to our mean departures
coeffOfSpatialDependency = np.nanmean(LISAlignedWeighted)
print( "Weighted coeff %2.8f" %coeffOfSpatialDependency)

LISGradient = np.gradient(LISAligned, axis=1)

# --- Plot data and create output ---
vmin, vmax = 5, 25
vadj = .101
cmap = plt.cm.twilight_shifted_r
cmap2 = plt.cm.bone_r
normRawData = colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=vmin, vmax=vmax)
normDep = colors.SymLogNorm(linthresh=0.1, linscale=1, vmin=np.nanmin(LISAlignedMeanDep), vmax=np.nanmax(LISAlignedMeanDep))
#normSlope = colors.LogNorm(vmin=0, vmax=np.nanmax(np.abs(LISGradient)))
extent = [-1, 1, -1, 1]
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(14, 12), constrained_layout=True,
                         gridspec_kw={'width_ratios': [10, 1, 10, 1], 'height_ratios': [10, 1, 10, 1], 'wspace': 0.0, 'hspace': 0.0})

#calculate the crds of our ROI for projecting the basemap
llcrnrlon= np.min(polyVerts[:,0] -.125)
llcrnrlat= np.min(polyVerts[:,1] -.125)
urcrnrlon= np.max(polyVerts[:,0] +.125)
urcrnrlat= np.max(polyVerts[:,1] +.125)

m = Basemap( projection='lcc', resolution='c', rsphere=(6378137.00,6356752.3142), 
			llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, llcrnrlat=llcrnrlat, 
			urcrnrlat=urcrnrlat, lat_1=33., lat_2=45., lat_0=39., lon_0=-96., ax=axes[0][0], fix_aspect=True)

# Entire roi:
#m = Basemap( projection='lcc', resolution='c', rsphere=(6378137.00,6356752.3142), 
#			llcrnrlon=-108.0, urcrnrlon=-92.50, llcrnrlat=38.0, 
#			urcrnrlat=50., lat_1=33., lat_2=45., lat_0=39., lon_0=-96., ax=axs[0])

#m.readshapefile('/mnt/d/Libraries/Documents/Scripts/LIS_Plot/Test_Data/Aux_files/States_21basic/states', 'states')
#m.readshapefile('/mnt/d/Libraries/Documents/Scripts/LIS_Plot/Test_Data/Aux_files/tl_2017_us_county/tl_2017_us_county','tl_2017_us_county')
LISPlotX, LISPlotY = m(LISGridLon, LISGridLat)
m.drawparallels(np.arange(-90.,120.,1),labels=[1,0,0,0])
m.drawmeridians(np.arange(-120.,-80.,1),labels=[0,0,0,1])
sp00 = m.pcolormesh(LISPlotX,LISPlotY,LISDataMasked,shading='flat', cmap=cmap, norm=normRawData)
UPlotX, UPlotY = m(UGridLon, UGridLat)
axes[0][0].barbs(UPlotX, UPlotY, U, V, pivot='tip', barbcolor='#333333')	
plt.colorbar(sp00, orientation='vertical', shrink=0.5, ax=axes[0][1])

#plot the distribution of RSM values
#axes[1][0].hist(LISAligned.flatten(), bins='fd')

# --- Plot wind orientated data with axis averages ---
z = LISAligned
z1 = np.nanmean(z, axis=1).reshape(z.shape[0], 1)
z0 = np.nanmean(z, axis=0).reshape(1, z.shape[1])
sp02 = axes[0][2].imshow(z, cmap=cmap, extent=extent, aspect=1, origin='lower', norm=normRawData)
axes[0][2].xaxis.tick_bottom()
axes[0][2].axhline(y=0, color='k')
axes[0][2].axvline(x=0, color='k')
plt.colorbar(sp02, orientation='vertical', shrink=0.5, ax=axes[0][3])

sp03 = axes[0][3].imshow(z1, cmap=cmap, extent=extent, aspect=10/1, origin='lower', norm=normRawData)
axes[0][3].set_xticks([])
axes[0][3].yaxis.tick_right()

sp12 = axes[1][2].imshow(z0, cmap=cmap, extent=extent, aspect=1/10, origin='lower', norm=normRawData)
axes[1][2].set_yticks([])

# --- Plot wind orientated data (variance) with axis averages ---
z = LISAlignedMeanDep
z1 = np.nanmean(z, axis=1).reshape(z.shape[0], 1)
z0 = np.nanmean(z, axis=0).reshape(1, z.shape[1])

sp20 = axes[2][0].imshow(z, cmap=cmap, extent=extent, aspect=1, origin='lower', norm=normDep)
axes[2][0].xaxis.tick_bottom()
axes[2][0].axhline(y=0, color='k')
axes[2][0].axvline(x=0, color='k')
plt.colorbar(sp20, orientation='vertical', shrink=0.5, ax=axes[2][1])

sp21 = axes[2][1].imshow(z1, cmap=cmap, extent=extent, aspect=10/1, origin='lower', norm=normDep)
axes[2][1].set_xticks([])
axes[2][1].yaxis.tick_right()

sp30 = axes[3][0].imshow(z0, cmap=cmap, extent=extent, aspect=1/10, origin='lower', norm=normDep)
axes[3][0].set_yticks([])

# --- Plot wind orientated data (variance gaussian weighted) with axis averages ---
z = LISGradient
z1 = np.nanmean(z, axis=1).reshape(z.shape[0], 1)
z0 = np.nanmean(z, axis=0).reshape(1, z.shape[1])
sp22 = axes[2][2].imshow(z, cmap=cmap2, extent=extent, aspect=1, origin='lower')
plt.colorbar(sp22, orientation='vertical', shrink=0.5, ax=axes[2][3])

axes[2][2].xaxis.tick_bottom()
axes[2][2].axhline(y=0, color='k')
axes[2][2].axvline(x=0, color='k')

sp23 = axes[2][3].imshow(z1, cmap=cmap2, extent=extent, aspect=10/1, origin='lower')
axes[2][3].set_xticks([])
axes[2][3].yaxis.tick_right()

sp32 = axes[3][2].imshow(z0, cmap=cmap2, extent=extent, aspect=1/10, origin='lower')
axes[3][2].set_yticks([])

#todo: o'all title       plt.title('Title')
axes[0][0].set_title('SPoRT LIS: RSM 0-10cm (%)' + os.linesep + \
						'Orientated W/ GFS4 10 m Winds' + os.linesep + \
						'Data Centered On: ' + str(args["lat_lon"]))
axes[0][2].set_title('RSM (%) - Surface Wind Orientated')
axes[0][2].set_ylabel('degrees downwind')
axes[0][2].set_xlabel('degrees cross-wind')
axes[0][3].set_ylabel('degrees downwind mean')
axes[1][2].set_xlabel('degrees cross-wind mean')
axes[1][0].set_ylabel('cells')
axes[2][0].set_title('Departures from Mean')
axes[2][2].set_title('Gradient (Slope)')

axes[0][1].axis('off')
axes[1][0].axis('off')		#hist of raw values
axes[1][1].axis('off')
axes[1][3].axis('off')
axes[3][1].axis('off')
axes[3][3].axis('off')

plt.savefig(args["output"] +'.png') # Set the output file name

# --- Save output as ascii format in .txt file ---
print("Saving .ascii ...")
ncols = LISAligned.shape[0]
nrows = LISAligned.shape[1]
cellsize = 3.0/111.0
xllcorner = -(LISAligned.shape[0]/2) * cellsize
yllcorner = -(LISAligned.shape[1]/2) * cellsize
nodata_value = -9999
np.nan_to_num(LISAligned, copy=False, nan=-9999)	
values = LISAligned.flatten()
toWrite = 'ncols ' + str(ncols) +\
			'\nnrows ' + str(nrows) +\
			'\nxllcorner ' + str(xllcorner) +\
			'\nyllcorner ' + str(yllcorner) +\
			'\ncellsize ' + str(cellsize) +\
			'\nnodata_value ' + str(nodata_value) + '\n' +\
			' '.join(map(str, values))
ascii_file = open(args["output"] +'.txt', "w")
ascii_file.write(toWrite)
ascii_file.close()
