#!/usr/bin/env python
# coding: utf-8

# # Example of preparing 2 image inputs for comparitive analysis 
# 1. Viewing input projections
# 2. warping input arrays to common grid
# 3. getting the common masked data (the union of each mask)
# 4. applying the common mask to the warped arrays  

# In[1]:


import os
import sys
import time
# Glenn's added dependencies
from datetime import datetime

import numpy as np
import osgeo
########################################
# Import python packages
########################################
import rasterio
from osgeo import gdal
from pygeotools.lib import iolib, malib, warplib
from sklearn.linear_model import LinearRegression

from srlite.model.PlotLib import PlotLib

# Specify regression algorithm
# regressionType = 'scipy'
regressionType = 'sklearn'

def getBandIndices(fn_list, bandNamePair):
    """
    TODO: Replace logic with real mapping...
    Temporary slop to deal with lack of metadata in WV files
    """
    ccdcDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
    ccdcBands = ccdcDs.RasterCount
    evhrDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)
    evhrBands = evhrDs.RasterCount

    pl.trace('bandNamePair: ' + str(bandNamePair))
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
#/att/nobackup/gtamkin/dev/srlite/input/2011/WV02_20110818/WV02_20110818-full-scene
# year = '2011'
# doy = year + '0818'
# WV02_20150616_M1BS_103001004351F000-toa.tif
year = '2015'
doy = year + '0616'

dataset = 'WV02_'
region = '-full-scene'
details = year + '/' + dataset + doy + '/' + dataset + doy + region
maindir = '/att/nobackup/gtamkin/dev/srlite'
input = maindir + '/input'
r_fn_ccdc = os.path.join(input + '/' + year + '/' + 'ccdc_' + doy + '-edited.tif')
r_fn_evhr = os.path.join(input + '/' + details, 'WV02_20150616_M1BS_103001004351F000-toa.tif')  # open for band descriptions
#r_fn_evhr = os.path.join(input + '/' + details, 'WV02_20110818_M1BS_103001000CCC9000-toa.tif')  # open for band descriptions
#r_fn_evhr = os.path.join(maindir, 'WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif')
#r_fn_evhr_full = os.path.join(maindir, '_WV02_20110818_M1BS_103001000CCC9000-toa.tif')  # open for band descriptions

# r_fn_ccdc = os.path.join(maindir, 'ccdc_20150616-edited.tif')
# r_fn_evhr = os.path.join(maindir, 'WV02_20150616_M1BS_103001004351F000-toa.tif')  # open for band descriptions
# #r_fn_evhr = os.path.join(maindir, 'WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif')
# r_fn_evhr_full = os.path.join(maindir, '_WV02_20150616_M1BS_103001004351F000-toa.tif')  # open for band descriptions

fn_list = [r_fn_ccdc, r_fn_evhr]
OUTDIR = input + '/' + details
#OUTDIR = maindir + '/output/' + details
# data_chunks = {'band': 1, 'x': 2048, 'y': 2048}

##############################################
# Default configuration values
# Debug levels:  0-no debug, 2-visualization, 3-detailed diagnostics
debugLevel = 3

# Toggle visualizations
imagePlot = False
histogramPlot = False
scatterPlot = False
fitPlot = False
# imagePlot = True
# histogramPlot = True
# scatterPlot = True
# fitPlot = True

override = False
pl = PlotLib(debugLevel, histogramPlot, scatterPlot, fitPlot)

pl.trace('\nCCDC file=' + r_fn_ccdc + '\nEVHR file=' + r_fn_evhr)

# Temporary input - Need to pull from arg list
bandNamePairList = list([
    ['blue', 'BAND-B'],
    ['green', 'BAND-G'],
    ['red', 'BAND-R'],
    ['nir', 'BAND-N']])

########################################
# Validate Band Pairs and Retrieve Corresponding Array Indices
########################################
pl.trace('bandNamePairList=' + str(bandNamePairList))
fn_full_list = [r_fn_ccdc, r_fn_evhr]
bandPairIndicesList = getBandIndices(fn_full_list, bandNamePairList)
pl.trace('bandIndices=' + str(bandPairIndicesList))

# ########################################
# # Align the CCDC and EVHR images, then take the intersection of the grid points
# ########################################
warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='average')
warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
ccdc_warp_ma, evhr_warp_ma = warp_ma_list

########################################
# ### FOR EACH BAND PAIR,
# now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
########################################
pl.trace('bandPairIndicesList: ' + str(bandPairIndicesList))
numBandPairs = len(bandPairIndicesList)
warp_ma_masked_band_series = [numBandPairs]
sr_prediction_list = [numBandPairs]
# debug_level = 3
for bandPairIndex in range(0, numBandPairs - 1):

    #    override = False
    bandPairIndices = bandPairIndicesList[bandPairIndex + 1]
    ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
    evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

    # suppress negative values in destination array
    ccdcBandMaArrayRaw = iolib.fn_getma(fn_list[0], bandPairIndices[0])
    evhrBandMaArrayRaw = iolib.fn_getma(fn_list[1], bandPairIndices[1])

    ########################################
    # Mask negative values in input prior to generating common mask ++++++++++[as per PM - 01/05/2022]
    ########################################
    warp_ma_band_list = [ccdcBandMaArray, evhrBandMaArray]
    warp_valid_ma_band_list = warp_ma_band_list
    common_mask_band = malib.common_mask(warp_valid_ma_band_list)

    warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band),
                                np.ma.array(evhrBandMaArray, mask=common_mask_band)]
    # Check the mins of each ma - they should be greater than 0
    for j, ma in enumerate(warp_ma_masked_band_list):
        j = j + 1
        if (ma.min() < 0):
            pl.trace("Warning: Masked array values should be larger than 0")
#            exit(1)
    pl.plot_maps(warp_ma_masked_band_list, fn_list, figsize=(10, 5),
                 title=str(bandNamePairList[bandPairIndex]) + ' Reflectance (%)')
    pl.plot_histograms(warp_ma_masked_band_list, fn_list, figsize=(10, 3),
                       title=str(bandNamePairList[bandPairIndex]) + " BAND COMMON ARRAY")

    warp_ma_masked_band_series.append(warp_ma_masked_band_list)

    ########################################
    # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
    # CCDC SR is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
    ########################################
    ccdc_sr_band = warp_ma_masked_band_list[0].ravel()
    evhr_toa_band = warp_ma_masked_band_list[1].ravel()

    ccdc_sr_data_only_band = ccdc_sr_band[ccdc_sr_band.mask == False]
    evhr_toa_data_only_band = evhr_toa_band[evhr_toa_band.mask == False]

    model_data_only_band = LinearRegression().fit(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
    pl.trace(str(bandNamePairList[bandPairIndex]) + '\nintercept: ' + str(
        model_data_only_band.intercept_) + ' slope: ' + str(model_data_only_band.coef_) + ' score: ' +
          str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)))
    pl.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                model_data_only_band.intercept_, override=override)

    ########################################
    # #### Apply the model to the original EVHR (2m) to predict surface reflectance
    ########################################
    pl.trace(f'Applying model to {str(bandNamePairList[bandPairIndex])} in file {os.path.basename(fn_list[1])}')
    pl.trace(f'Input masked array shape: {evhrBandMaArray.shape}')

    sr_prediction_band = model_data_only_band.predict(evhrBandMaArrayRaw.ravel().reshape(-1, 1))
    pl.trace(f'Post-prediction shape : {sr_prediction_band.shape}')

    # Return to original shape and apply original mask
    orig_dims = evhrBandMaArrayRaw.shape
    evhr_sr_ma_band = np.ma.array(sr_prediction_band.reshape(orig_dims), mask=evhrBandMaArrayRaw.mask)

     # Check resulting ma
    pl.trace(f'Final masked array shape: {evhr_sr_ma_band.shape}')
    pl.trace('evhr_sr_ma=\n' + str(evhr_sr_ma_band))

    ########### save prediction #############
    sr_prediction_list.append(evhr_sr_ma_band)

    ########################################
    ##### Compare the before and after histograms (EVHR TOA vs EVHR SR)
    ########################################
    evhr_pre_post_ma_list = [evhrBandMaArrayRaw, evhr_sr_ma_band]
    compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']

    pl.plot_histograms(evhr_pre_post_ma_list, fn_list, figsize=(5, 3),
                       title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR", override=override)
    pl.plot_maps(evhr_pre_post_ma_list, compare_name_list, figsize=(10, 50), override=override)

    ########################################
    ##### Compare the original CCDC histogram with result (CCDC SR vs EVHR SR)
    ########################################
    ccdc_evhr_srlite_list = [ccdc_warp_ma, evhr_sr_ma_band]
    compare_name_list = ['CCDC SR', 'EVHR SR-Lite']

    pl.plot_histograms(ccdc_evhr_srlite_list, fn_list, figsize=(5, 3),
                       title=str(bandNamePairList[bandPairIndex]) + " CCDC SR vs EVHR SR", override=override)
    pl.plot_maps(ccdc_evhr_srlite_list, compare_name_list, figsize=(10, 50), override=override)

    ########################################
    ##### Compare the original EVHR TOA histogram with result (EVHR TOA vs EVHR SR)
    ########################################
    evhr_srlite_delta_list = [evhr_pre_post_ma_list[1], evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]]
    compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']
    pl.plot_histograms(evhr_srlite_delta_list, fn_list, figsize=(5, 3),
                       title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR DELTA ", override=override)
    pl.plot_maps([evhr_pre_post_ma_list[1],
                  evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]],
                 [compare_name_list[1],
                  str(bandNamePairList[bandPairIndex]) + ' Difference: TOA-SR-Lite'], (10, 50),
                 cmap_list=['RdYlGn', 'RdBu'], override=override)
    print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

########################################
# Create .tif image from band-based prediction layers
########################################
print(f"\nApply coefficients to High Res File...{r_fn_evhr}")

debug_level = 3
now = datetime.now()  # current date and time
# nowStr = now.strftime("%b:%d:%H:%M:%S")
nowStr = now.strftime("%m%d%H%M")
# print("time:", nowStr)
#  Derive file names for intermediate files
head, tail = os.path.split(r_fn_evhr)
filename = (tail.rsplit(".", 1)[0])
output_name = "{}/srlite-{}-{}".format(
    OUTDIR, filename,
    nowStr, r_fn_evhr.split('/')[-1]
) + ".tif"

# Read metadata of first file
with rasterio.open(r_fn_evhr) as src0:
    meta = src0.meta

# Update meta to reflect the number of layers
meta.update(count=numBandPairs-1)

# Read each layer and write it to stack
with rasterio.open(output_name, 'w', **meta) as dst:
    for id in range(1, numBandPairs):
        bandPrediction = sr_prediction_list[id]
        min = bandPrediction.min()
        pl.trace(f'BandPrediction.min({min})')
        dst.set_band_description(id, bandNamePairList[id - 1][1])
        bandPrediction1 = np.ma.masked_values(bandPrediction, -9999)
        dst.write_band(id, bandPrediction1)

print("Elapsed Time: " + output_name + ': ',
          (time.time() - start_time) / 60.0)  # time in min
