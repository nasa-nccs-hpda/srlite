#!/usr/bin/env python
# coding: utf-8

# # Example of preparing 2 image inputs for comparitive analysis 
# 1. Viewing input projections
# 2. warping input arrays to common grid
# 3. getting the common masked data (the union of each mask)
# 4. applying the common mask to the warped arrays  

# In[1]:


########################################
# Import python packages
########################################
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import geopandas
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.colors import LinearSegmentedColormap000
#from matplotlib.colors import Normalize


import pandas as pd
import numpy as np
import numpy.ma as ma

# Glenn's added dependencies
from scipy import stats
import xarray as xr  # array manipulation library, rasterio built-in
import rasterio as rio  # geospatial library
import dask
from datetime import datetime 
from rasterio.plot import show_hist
from matplotlib import pyplot
from time import time  # tracking time
from sklearn.linear_model import LinearRegression
from PIL import Image
import sys, time 
from pygeotools.lib import iolib, malib, geolib, filtlib, warplib
import osgeo
from osgeo import gdal, ogr, osr
from plotnine import ggplot, aes, geom_line, geom_point, geom_smooth, geom_bin2d, geom_abline, xlim, ylim
import shutil

##############################################
# Default configuration values
# Debug levels:  0-no debug, 2-visualization, 3-detailed diagnostics
debug_level = 0

# Toggle visualizations
imagePlot = False
histogramPlot = True
scatterPlot = False
fitPlot = True
# imagePlot = True
# histogramPlot = True
# scatterPlot = True
# fitPlot = True

override = False

# Specify regression algorithm
#regressionType = 'scipy'
regressionType = 'sklearn'

# Print trace debug
def trace(value, override=False):
    if ((debug_level >= 3) or override==True):
        print(value)

trace('osgeo.gdal.VersionInfo()=' + str(osgeo.gdal.VersionInfo()))
trace('sys.path=' + str(sys.path))

# #### Functions

def plot_maps(masked_array_list, fn_list, figsize=(10,5), title='Reflectance (%)',
              cmap_list=['RdYlGn','RdYlGn'], override=False):

    if (((debug_level >= 2) and (imagePlot==True)) or override==True):
        fig, axa = plt.subplots( nrows=1, ncols=len(fn_list), figsize=figsize, sharex=False, sharey=False)
        for i, ma in enumerate(masked_array_list):

            f_name = fn_list[i]
            divider = make_axes_locatable(axa[i])
            cax = divider.append_axes('right', size='2.5%', pad=0.05)
            im1 = axa[i].imshow(ma, cmap=cmap_list[i] , clim=malib.calcperc(ma, perc=(1,95)) )
            cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend='max')
            axa[i].set_title(os.path.split(f_name)[1], fontsize=10)
            cb.set_label(title)

            plt.tight_layout()

def plot_histograms(masked_array_list, fn_list, figsize=(10,3), title="WARPED MASKED ARRAY", override=False):

    if (((debug_level >= 2) and (histogramPlot==True)) or override==True):
        fig, axa = plt.subplots(nrows=1, ncols=len(masked_array_list), figsize=figsize, sharex=True, sharey=True)

        for i, ma in enumerate(masked_array_list):
            f_name = os.path.split(fn_list[i])[1]
            trace(f" {ma.count()} valid pixels in {title} version of {f_name}")

            h = axa[i].hist(ma.compressed(), bins=512, alpha=0.75)
            axa[i].set_title(title + ' ' + f_name, fontsize=10)

        plt.tight_layout()

def plot_scatter(x_data,y_data, title="Raster Data Scatter Plot", null_value=-10, override=False):

    if (((debug_level >= 2) and (scatterPlot==True)) or override==True):
        plt.rcParams["font.family"] = "Times New Roman"
        #Declaring the figure, and hiding the ticks' labels
        fig, ax = plt.subplots(figsize=(15,8))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #Actually Plotting the data
        plt.scatter(x_data,y_data, s=0.1, c='black')
        #Making the graph pretty and informative!
        plt.title(title, fontsize=28)
        plt.xlabel("X-Axis Raster", fontsize=22)
        plt.ylabel("Y-Axis Raster", fontsize=22)
        plt.show()

# Plot Linear Fit
def plot_fit(x, y, slope, intercept, override=False):
    if (((debug_level >= 2) and (fitPlot==True)) or override==True):
        print(ggplot()  # What data to use
              # + aes(x="date", y="pop")  # What variable to use
              + aes(x=x, y=y)  # What variable to use
              + geom_bin2d(binwidth=10)  # Geometric object to use for drawing
              + geom_abline(slope=slope, intercept=intercept, size=2)
              + geom_smooth(method = 'lm', color='red'))
        # + xlim(0,500) + ylim(0,500)

def getBandIndices(fn_list, bandNamePair) :
    """
    TODO: Replace logic with real mapping...
    Temporary slop to deal with lack of metadata in WV files
    """
    ccdcDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
    ccdcBands = ccdcDs.RasterCount
    evhrDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)
    evhrBands = evhrDs.RasterCount

    trace('bandNamePair: ' + str(bandNamePair))
    numBandPairs = len(bandNamePair)
    bandIndices = [numBandPairs]

    for bandPairIndex in range(0, numBandPairs):

        ccdcBandIndex = evhrBandIndex = -1
        currentBandPair = bandNamePair[bandPairIndex]

        for ccdcIndex in range(1, ccdcBands + 1):
            # read in bands from image
            band = ccdcDs.GetRasterBand(ccdcIndex)
            bandDescription = band.GetDescription()
            bandName = currentBandPair[0]
            if (bandDescription == bandName):
                ccdcBandIndex = ccdcIndex
                break

        for evhrIndex in range(1, evhrBands + 1):
            # read in bands from image
            band = evhrDs.GetRasterBand(evhrIndex)
            bandDescription = band.GetDescription()
            bandName = currentBandPair[1]
            if (bandDescription == bandName):
                evhrBandIndex = evhrIndex
                break

        if ((ccdcBandIndex == -1) or (evhrBandIndex == -1)):
            ccdcDs = evhrDs = None
            print(f"Invalid band pairs - verify correct name and case {currentBandPair}")
            exit(1)

        bandIndices.append([ccdcIndex, evhrIndex])

    ccdcDs = evhrDs = None
    return bandIndices

########################################
# For this test run, use a subset of the YKD (e.g., Pitkus Point) for faster processing time
########################################
start_time = time.time()  # record start time

maindir = '/home/centos/srlite/pitkus-point-demo/02-07-2022/'
r_fn_ccdc = os.path.join(maindir, 'ccdc_20110818-edited.tif')
r_fn_evhr_full = os.path.join(maindir, 'WV02_20110818_M1BS_103001000CCC9000-toa.tif') # open for band descriptions
r_fn_evhr = os.path.join(maindir, 'WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif')
fn_list = [r_fn_ccdc, r_fn_evhr]
OUTDIR = maindir + 'results'
#data_chunks = {'band': 1, 'x': 2048, 'y': 2048}

trace('\nCCDC file=' + r_fn_ccdc + '\nEVHR file=' + r_fn_evhr)

# Temporary input - Need to pull from arg list
bandNamePairList =  list([
    ['blue', 'BAND-B'],
    ['green', 'BAND-G'],
    ['red', 'BAND-R'],
    ['nir', 'BAND-N']])

########################################
# Validate Band Pairs and Retrieve Corresponding Array Indices
########################################
trace('bandNamePairList=' + str(bandNamePairList))
fn_full_list = [r_fn_ccdc, r_fn_evhr_full]
bandPairIndicesList = getBandIndices(fn_full_list, bandNamePairList)
trace('bandIndices=' + str(bandPairIndicesList))

########################################
# Get Masked Arrays for CCDC and EVHR file names
########################################
ma_list = [iolib.fn_getma(fn) for fn in fn_list]
plot_maps(ma_list, fn_list, figsize=(10,5))
plot_histograms(ma_list, fn_list, figsize=(10,3), title="INPUT MASKED ARRAY")

########################################
# Align the CCDC and EVHR images, then take the intersection of the grid points
########################################
warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='average')
warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
trace('CCDC shape=' + str(warp_ma_list[0].shape) + ' EVHR shape=' + str(warp_ma_list[1].shape))
plot_maps(warp_ma_list, fn_list, figsize=(10,5))
plot_histograms(warp_ma_list, fn_list, figsize=(10,3), title="INTERSECTION ARRAY")

########################################
# Mask negative values in input prior to generating common mask ++++++++++[as per PM - 01/05/2022]
########################################
warp_valid_ma_list = warp_ma_list
warp_valid_ma_list = [ np.ma.masked_where(ma < 0, ma) for ma in warp_ma_list]
common_mask = malib.common_mask(warp_valid_ma_list)
trace('common_mask='+ str(common_mask))

########################################
# ### WARPED MASKED ARRAY WITH COMMON MASK APPLIED
# now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
########################################
ccdc_warp_ma, evhr_warp_ma = warp_ma_list
warp_ma_masked_list = [np.ma.array(ccdc_warp_ma, mask=common_mask), np.ma.array(evhr_warp_ma, mask=common_mask)]
trace('warp_ma_masked_list='+ str(warp_ma_masked_list))
# Check the mins of each ma - they should be greater than 0
for i, ma in enumerate(warp_ma_masked_list):
    if (ma.min() < 0):
        trace("Masked array values should be larger than 0")
        exit(1)
plot_maps(warp_ma_masked_list, fn_list, figsize=(10,5))
plot_histograms(warp_ma_masked_list, fn_list, figsize=(10,3), title="COMMON ARRAY")

########################################
# ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
# CCDC SR is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
########################################
ccdc_sr = warp_ma_masked_list[0].ravel()
evhr_toa = warp_ma_masked_list[1].ravel()

ccdc_sr_data_only = ccdc_sr[ccdc_sr.mask == False]
evhr_toa_data_only = evhr_toa[evhr_toa.mask == False]
evhr_toa_data_only_reshaped = evhr_toa_data_only.reshape(-1,1)
model_data_only = LinearRegression().fit(evhr_toa_data_only.reshape(-1,1), ccdc_sr_data_only)
trace('intercept: ' + str(model_data_only.intercept_) + ' slope: ' + str(model_data_only.coef_) + ' score: ' +
      str(model_data_only.score(evhr_toa_data_only.reshape(-1,1), ccdc_sr_data_only )))
plot_fit(evhr_toa_data_only, ccdc_sr_data_only, model_data_only.coef_[0], model_data_only.intercept_)

########################################
# #### Apply the model to the original EVHR (2m) to predict surface reflectance
########################################
trace(f'Applying model to {os.path.basename(fn_list[1])}')
trace(f'Input masked array shape: {ma_list[1].shape}')

sr_prediction = model_data_only.predict(ma_list[1].ravel().reshape(-1,1))
trace(f'Post-prediction shape: {sr_prediction.shape}')

# Return to original shape and apply original mask
orig_dims = ma_list[1].shape
evhr_sr_ma = np.ma.array(sr_prediction.reshape(orig_dims), mask=ma_list[1].mask)

# Check resulting ma
trace(f'Final masked array shape: {evhr_sr_ma.shape}')
trace('evhr_sr_ma=\n' + str(evhr_sr_ma))

########################################
##### Compare the before and after histograms (EVHR TOA vs EVHR SR)
########################################
evhr_pre_post_ma_list = [ma_list[1], evhr_sr_ma]
compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']

plot_histograms(evhr_pre_post_ma_list, fn_list, figsize=(5,3), title="EVHR TOA vs EVHR SR")
plot_maps(evhr_pre_post_ma_list, compare_name_list, (10,50))

########################################
##### Compare the original CCDC histogram with result (CCDC SR vs EVHR SR)
########################################
ccdc_evhr_srlite_list = [ccdc_warp_ma, evhr_sr_ma]
compare_name_list = ['CCDC SR', 'EVHR SR-Lite']

plot_histograms(ccdc_evhr_srlite_list, fn_list, figsize=(5,3), title="CCDC SR vs EVHR SR")
plot_maps(ccdc_evhr_srlite_list, compare_name_list, figsize=(10,50))

########################################
##### Compare the original EVHR TOA historgram with result (EVHR TOA vs EVHR SR)
########################################
evhr_srlite_delta_list = [evhr_pre_post_ma_list[1], evhr_pre_post_ma_list[1]-evhr_pre_post_ma_list[0]]
compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']
plot_histograms(evhr_srlite_delta_list, fn_list, figsize=(5,3), title="EVHR TOA vs EVHR SR")
plot_maps([evhr_pre_post_ma_list[1], evhr_pre_post_ma_list[1]-evhr_pre_post_ma_list[0]], [compare_name_list[1], 'Difference: TOA-SR-Lite'], (10,50), cmap_list=['RdYlGn','RdBu'])


########################################
# ### FOR EACH BAND PAIR,
# now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
########################################

################
##### Get Band Indices for Pair
################

trace('bandPairIndicesList: ' + str(bandPairIndicesList))
numBandPairs = len(bandPairIndicesList)
warp_ma_masked_band_series = [numBandPairs]
sr_prediction_list = [numBandPairs]
#debug_level = 3
for bandPairIndex in range(0, numBandPairs-1):

#    override = False
    bandPairIndices = bandPairIndicesList[bandPairIndex+1]
    ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
    evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

    # suppress negative values in destination array
    ccdcBandMaArrayRaw = iolib.fn_getma(fn_list[0], bandPairIndices[0])
    evhrBandMaArrayRaw = iolib.fn_getma(fn_list[1], bandPairIndices[1])
    # ccdcBandMaArrayRaw = np.ma.masked_where(ccdcBandMaArrayRaw < 0, ccdcBandMaArrayRaw)
    # evhrBandMaArrayRaw = np.ma.masked_where(evhrBandMaArrayRaw < 0, evhrBandMaArrayRaw)

    ########################################
    # Mask negative values in input prior to generating common mask ++++++++++[as per PM - 01/05/2022]
    ########################################
    warp_ma_band_list = [ccdcBandMaArray, evhrBandMaArray]
    warp_valid_ma_band_list = warp_ma_band_list
#    warp_valid_ma_band_list = [np.ma.masked_where(ma < 0, ma) for ma in warp_ma_band_list]
    common_mask_band = malib.common_mask(warp_valid_ma_band_list)
#    trace('common_mask_band=' + str(common_mask_band))

    warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band), np.ma.array(evhrBandMaArray, mask=common_mask_band)]
#    trace('warp_ma_masked_band_list=' + str(warp_ma_masked_band_list))
    # Check the mins of each ma - they should be greater than 0
    for j, ma in enumerate(warp_ma_masked_band_list):
        j = j + 1
        if (ma.min() < 0):
            trace("Masked array values should be larger than 0")
            exit(1)
    plot_maps(warp_ma_masked_band_list, fn_list, figsize=(10, 5), title=str(bandNamePairList[bandPairIndex]) + ' Reflectance (%)')
    plot_histograms(warp_ma_masked_band_list, fn_list, figsize=(10, 3), title=str(bandNamePairList[bandPairIndex]) + " BAND COMMON ARRAY")

    warp_ma_masked_band_series.append(warp_ma_masked_band_list)

    ########################################
    # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
    # CCDC SR is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
    ########################################
    ccdc_sr_band = warp_ma_masked_band_list[0].ravel()
    evhr_toa_band = warp_ma_masked_band_list[1].ravel()

    ccdc_sr_data_only_band = ccdc_sr_band[ccdc_sr_band .mask == False]
    evhr_toa_data_only_band = evhr_toa_band[evhr_toa_band .mask == False]

    model_data_only_band = LinearRegression().fit(evhr_toa_data_only_band .reshape(-1,1), ccdc_sr_data_only_band )
    trace(str(bandNamePairList[bandPairIndex])+'\nintercept: ' + str(model_data_only_band.intercept_) + ' slope: ' + str(model_data_only.coef_) + ' score: ' +
          str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1,1), ccdc_sr_data_only_band )))
    plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0], model_data_only_band.intercept_, override=override)

    ########################################
    # #### Apply the model to the original EVHR (2m) to predict surface reflectance
    ########################################
    trace(f'Applying model to {str(bandNamePairList[bandPairIndex])} in file {os.path.basename(fn_list[1])}')
    trace(f'Input masked array shape: {evhrBandMaArray.shape}')

#    evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])
    sr_prediction_band = model_data_only.predict(evhrBandMaArrayRaw.ravel().reshape(-1,1))
#    sr_prediction_band = np.ma.masked_where(sr_prediction_band < 0, sr_prediction_band)
#    sr_prediction_band = model_data_only.predict(evhrBandMaArray.ravel().reshape(-1,1))
    trace(f'Post-prediction shape : {sr_prediction_band.shape}')

    # Return to original shape and apply original mask
    orig_dims = evhrBandMaArrayRaw.shape
    evhr_sr_ma_band = np.ma.array(sr_prediction_band.reshape(orig_dims), mask=evhrBandMaArrayRaw.mask)
#    evhr_sr_ma_band = np.ma.masked_where(evhr_sr_ma_band < 0, evhr_sr_ma_band)
#    sr_prediction_band = np.ma.masked_where(sr_prediction_band < 0, sr_prediction_band)

    ########### save prdection #############
    #sr_prediction_list.append(sr_prediction)

    # Check resulting ma
    trace(f'Final masked array shape: {evhr_sr_ma_band.shape}')
    trace('evhr_sr_ma=\n' + str(evhr_sr_ma_band))

    ########### save prdection #############
    sr_prediction_list.append(evhr_sr_ma_band)

    ########################################
    ##### Compare the before and after histograms (EVHR TOA vs EVHR SR)
    ########################################
#    override = True
    evhr_pre_post_ma_list = [evhrBandMaArrayRaw, evhr_sr_ma_band]
    compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']

    plot_histograms(evhr_pre_post_ma_list, fn_list, figsize=(5,3), title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR", override=override)
    plot_maps(evhr_pre_post_ma_list, compare_name_list, figsize=(10,50), override=override)

    ########################################
    ##### Compare the original CCDC histogram with result (CCDC SR vs EVHR SR)
    ########################################
    ccdc_evhr_srlite_list = [ccdc_warp_ma, evhr_sr_ma_band]
    compare_name_list = ['CCDC SR', 'EVHR SR-Lite']

    plot_histograms(ccdc_evhr_srlite_list, fn_list, figsize=(5,3), title=str(bandNamePairList[bandPairIndex]) + " CCDC SR vs EVHR SR", override=override)
    plot_maps(ccdc_evhr_srlite_list, compare_name_list, figsize=(10,50), override=override)

    ########################################
    ##### Compare the original EVHR TOA histogram with result (EVHR TOA vs EVHR SR)
    ########################################
    evhr_srlite_delta_list = [evhr_pre_post_ma_list[1], evhr_pre_post_ma_list[1]-evhr_pre_post_ma_list[0]]
    compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']
    plot_histograms(evhr_srlite_delta_list, fn_list, figsize=(5,3), title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR DELTA ", override=override)
    plot_maps([evhr_pre_post_ma_list[1],
               evhr_pre_post_ma_list[1]-evhr_pre_post_ma_list[0]],
              [compare_name_list[1],
              str(bandNamePairList[bandPairIndex]) + ' Difference: TOA-SR-Lite'], (10,50), cmap_list=['RdYlGn','RdBu'], override=override)
    print(f"On to Next Band")

########################################
# Open the High Resolution 2m EVHR raster
########################################
print(f"Apply coefficients...{r_fn_evhr}")
#        self.readraster(rast, args.bands_data)  # read raster

data_chunks = {'band': 1, 'x': 2048, 'y': 2048}
#data = xr.open_rasterio(r_fn_evhr, chunks=data_chunks)
_bandNames = ['b1', 'b2', 'b3', 'b4']
bandNames = ['blue', 'green', 'red', 'nir']
nodataval = (-9999.0, -9999.0, -9999.0, -9999.0)

########################################
# Transform the existing bands with the linear regression coefficients
########################################
def insertPredictionBands(data, sr_prediction_list, numBands):
        """
        """
        if (debug_level >= 2): print(f'\nNumBands={numBands} ')
        band = 0
        # Loop through the dataset and modify each band in order (assumes EVHR & CCDC bands align by default)
        while band < numBands:

            # # Retrieve the coefficients calculated above
            # slope, yInt = coefficients[band+1][0], coefficients[band+1][1]
            # if (debug_level >= 2): print ("BandIndex Slope Yint: ", band, slope, yInt)
            #
            #
            # ######################################
            # ######### mx + b #####################
            # data[band]  = (data[band] * slope) + yInt
            # ######### mx + b #####################
            # ######################################
            #
            # # Suppress the NoData pixels with the common mask
            # evhrCommonBandMaArray = dask.array.ma.masked_array(data[band], mask=np.ma.getmask(common_warp_ma_masked_evhr2m))

            evhrCommonBandMaArrayData = np.ma.getdata(sr_prediction_list[band])
#            evhrCommonBandMaArrayData = np.ma.masked_where(evhrCommonBandMaArrayData < 0, evhrCommonBandMaArrayData)
            data[band]  = evhrCommonBandMaArrayData
            band += 1

        # update raster metadata, xarray attributes
        data.attrs['scales'] = [data.attrs['scales'][0]] * numBands
        data.attrs['offsets'] = [data.attrs['offsets'][0]] * numBands

        return data

########################################
# Apply the model
########################################
# add transformed WV Bands
# B1(Blue): 450 - 510
# B2(Green): 510 - 580
# B3(Red): 655 - 690
# B4(NIR): 780 - 920
data_chunks = {'band': 1, 'x': 2048, 'y': 2048}
#hrData = xr.open_rasterio(r_fn_evhr, chunks=data_chunks)

#ccdcNumBands = ccdc_warp_ds.RasterCount
#xFormedData = xformBands(data=hrData, numBands=ccdcNumBands, coefficients=coefficients)
#xFormedData = insertPredictionBands(hrData, sr_prediction_list, numBandPairs-1)

############## RASTERIO GROUND UP FILE CREATION ############
########################################
# Prep for output
########################################
debug_level = 3
now = datetime.now()  # current date and time
nowStr = now.strftime("%b:%d:%H:%M:%S")
#nowStr = now.strftime("%H:%M:%S")
# print("time:", nowStr)
#  Derive file names for intermediate files
head, tail = os.path.split(r_fn_evhr)
filename = (tail.rsplit(".", 1)[0])
output_name = "{}/{}-{}".format(
    OUTDIR, filename+'-srlite',
    nowStr, r_fn_evhr.split('/')[-1]
) + ".tif"

# Read metadata of first file
with rasterio.open(r_fn_evhr) as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count=4)

# Read each layer and write it to stack
with rasterio.open(output_name, 'w', **meta) as dst:
    # for id, layer in enumerate(sr_prediction_list, start=1):
    #     with rasterio.open(layer) as src1:
    #         dst.write_band(id, src1.read(1))

    for id in range(1,5):
#       #buid each year stack file
#     band=outds.GetRasterBand(index)
#     ndv = band.GetNoDataValue()
#     desc = band.GetDescription()
#     newBandName = bandNamePairList[index-1][1]
#     trace(f'\nBand #{index}: ndv=({ndv}), old name=({desc}), new name=({newBandName})')
# #    arr=band.ReadAsArray(sr_prediction_list[index])
# #    outds.GetRasterBand(index).WriteArray(arr)
        bandPrediction = sr_prediction_list[id]
        min = bandPrediction.min()
        trace(f'BandPrediction({bandPrediction}) \nBandPrediction.min({min})')
        dst.set_band_description(id, bandNamePairList[id-1][1])
        bandPrediction1 = np.ma.masked_values(bandPrediction, -9999)
#        bandPrediction1 = np.ma.masked_where(bandPrediction > 0, bandPrediction)
        dst.write_band(id, bandPrediction1)
exit()
#
# with rasterio.open(r_fn_evhr) as src_dataset:
#
#     # Get a copy of the source dataset's profile. Thus our
#     # destination dataset will have the same dimensions,
#     # number of bands, data type, and georeferencing as the
#     # source dataset.
#     kwds = src_dataset.profile
#
#     # Change the format driver for the destination dataset to
#     # 'GTiff', short for GeoTIFF.
#     kwds['driver'] = 'GTiff'
#
#     # Add GeoTIFF-specific keyword arguments.
#     kwds['tiled'] = True
#     kwds['blockxsize'] = 256
#     kwds['blockysize'] = 256
#     kwds['photometric'] = 'YCbCr'
#     kwds['compress'] = 'JPEG'
#
#     with rasterio.open(output_name, 'w', **kwds) as dst_dataset:
#         # Write data to the destination dataset.
#
#
############## DIRECT GDAL COPY/MODIFY ####################
########################################
# Prep for output
########################################
now = datetime.now()  # current date and time
nowStr = now.strftime("%b:%d:%H:%M:%S")
# print("time:", nowStr)
#  Derive file names for intermediate files
head, tail = os.path.split(r_fn_evhr)
filename = (tail.rsplit(".", 1)[0])
output_name = "{}/srlite-{}-{}".format(
    OUTDIR, filename,
    nowStr, r_fn_evhr.split('/')[-1]
) + ".tif"

driver = gdal.GetDriverByName('GTiff')
driver.Register()
data0=gdal.Open(r_fn_evhr, gdal.GA_ReadOnly)  # base file for driver.CreatCopy()
outfile = os.path.join(OUTDIR, output_name)
outds = driver.CreateCopy(outfile, data0)
debug_level = 3

for index in range(1,5):
      #buid each year stack file
    band=outds.GetRasterBand(index)
    ndv = band.GetNoDataValue()
    desc = band.GetDescription()
    newBandName = bandNamePairList[index-1][1]
    trace(f'\nBand #{index}: ndv=({ndv}), old name=({desc}), new name=({newBandName})')
#    arr=band.ReadAsArray(sr_prediction_list[index])
#    outds.GetRasterBand(index).WriteArray(arr)
    bandPrediction = sr_prediction_list[index]
    min = bandPrediction.min()
    trace(f'BandPrediction({bandPrediction}) \nBandPrediction.min({min})')

    # bandPredictionPositiveValuesOnly = np.ma.masked_where(bandPrediction < 0, bandPrediction)
    # min = bandPredictionPositiveValuesOnly.min()
    # trace(f'bandPredictionPositiveValuesOnly({bandPredictionPositiveValuesOnly}) \nbandPredictionPositiveValuesOnly.min({min})')

    # bandPredictionValidValuesOnly = bandPrediction[~bandPrediction.mask]
    # min = bandPredictionValidValuesOnly.min()
    # trace(f'bandPredictionValidValuesOnly({bandPredictionValidValuesOnly}) \nbandPredictionValidValuesOnly.min({min})')

    band.SetNoDataValue(ndv)
    band.SetDescription(newBandName) # This sets the band name!
    # if index == 2:
    #     SizeX = 12615
    #     SizeY = 6855
    #     outds.GetRasterBand(index).WriteArray(np.ones((SizeX, SizeY)))
    # else:
    band.WriteArray(bandPrediction)
    #band.WriteArray(bandPredictionPositiveValuesOnly)
#        outds.GetRasterBand(index).WriteArray(bandPredictionValidValuesOnly)

    # predictionlist = [bandPredictionValidValuesOnly, bandPredictionValidValuesOnly]
    # plot_histograms(predictionlist, fn_list, figsize=(5,3), title=" band #"+ str(index) + " bandPredictionValidValuesOnly ", override=True)

    predictionlist = [bandPrediction, bandPrediction]
    plot_histograms(predictionlist, fn_list, figsize=(5,3), title=" band #"+ str(index) + " bandPredictionPositiveValuesOn ", override=True)

    # predictionlist = [bandPredictionPositiveValuesOnly, bandPredictionPositiveValuesOnly]
    # plot_histograms(predictionlist, fn_list, figsize=(5,3), title=" band #"+ str(index) + " bandPredictionPositiveValuesOn ", override=True)

data0 = None
exit()

########################################
# Write each output band to disk
########################################
def torasterBands(rast, data, output='rfmask.tif'):
    """
    :param rast: raster name to get metadata from
    :param band: numpy array with synthetic band output
    :param output: raster name to save on
    :return: tif file saved to disk
    """
    currentNumBands = xFormedData.shape[0]

    # get meta features from raster
    with rio.open(rast) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')

    # PM - Not sure about this...
    nodatavals[nodatavals == -9999] = nodataval[0]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 4  # output is four bands
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(output, 'w', **out_meta) as dst:
        index = 0
        while index < currentNumBands:
            bandArray = data[index]
            nbandArray = bandArray.as_numpy()

            if (debug_level >= 2): print(
                f'nbandArray.index =  {index} index[0][0] {nbandArray.values[0][0]} index[500][500] {nbandArray.values[500][500]}')
            dst.write(bandArray, index + 1)
            index += 1

    if (debug_level >= 2): print(f'Band saved at {output}')


########################################
# Write output file to disk
########################################
torasterBands(r_fn_evhr, xFormedData, output_name)

########################################
# Transform the existing bands with the linear regression coefficients
########################################
def xformBands(data, numBands, coefficients):
        """
        """
        if (debug_level >= 2): print(f'\nNumBands={numBands} ')
        band = 0
        # Loop through the dataset and modify each band in order (assumes EVHR & CCDC bands align by default)
        while band < numBands:

            # Retrieve the coefficients calculated above
            slope, yInt = coefficients[band+1][0], coefficients[band+1][1]
            if (debug_level >= 2): print ("BandIndex Slope Yint: ", band, slope, yInt)


            ######################################
            ######### mx + b #####################
            data[band]  = (data[band] * slope) + yInt
            ######### mx + b #####################
            ######################################

            # Suppress the NoData pixels with the common mask
            evhrCommonBandMaArray = dask.array.ma.masked_array(data[band], mask=np.ma.getmask(common_warp_ma_masked_evhr2m))
            evhrCommonBandMaArrayData = np.ma.getdata(evhrCommonBandMaArray)
            data[band]  = evhrCommonBandMaArrayData
            band += 1

        # update raster metadata, xarray attributes
        data.attrs['scales'] = [data.attrs['scales'][0]] * numBands
        data.attrs['offsets'] = [data.attrs['offsets'][0]] * numBands

        return data
