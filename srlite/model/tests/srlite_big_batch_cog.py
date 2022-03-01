#!/usr/bin/env python
# coding: utf-8

# # Example of preparing 2 image inputs for comparitive analysis 
# 1. Viewing input projections
# 2. warping input arrays to common grid
# 3. getting the common masked data (the union of each mask)
# 4. applying the common mask to the warped arrays  

# In[1]:


########################################
# Point to local pygeotools (not in ilab-kernel by default)
########################################
import sys
#sys.path.append('/att/gpfsfs/home/pmontesa/code/pygeotools')
sys.path.append('/home/gtamkin/.local/lib/python3.9/site-packages')
sys.path.append('/att/nobackup/gtamkin/dev/srlite/src')

debug_level = 0

import subprocess

#from core.model.SystemCommand import SystemCommand

# ## Change os.path.append() to your personal repo location

# In[2]:


if (debug_level >= 2):  print(sys.path)


# In[3]:


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
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Specify regression algorithm
# regressionType = 'scipy'
regressionType = 'sklearn'


# In[4]:


from pygeotools.lib import iolib, malib, geolib, filtlib, warplib


# In[5]:


import osgeo
from osgeo import gdal, ogr, osr
if (debug_level >= 2):  print(osgeo.gdal.VersionInfo())


# #### Functions

# In[6]:


def getBandIndices(fn_list, bandNamePair):
    """
    Validate band name pairs and return corresponding gdal indices
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


# In[7]:


def processBands(warp_ds_list, bandPairIndicesList, fn_list):

    ########################################
    # ### FOR EACH BAND PAIR,
    # now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
    ########################################
    from sklearn.linear_model import HuberRegressor
    pl.trace('bandPairIndicesList: ' + str(bandPairIndicesList))
    numBandPairs = len(bandPairIndicesList)
    warp_ma_masked_band_series = [numBandPairs]
    sr_prediction_list = [numBandPairs]
    # debug_level = 3
    for bandPairIndex in range(0, numBandPairs - 1):

        pl.trace('=>')
        pl.trace('====================================================================================')
        pl.trace('============== Start Processing Band #' + str(bandPairIndex + 1) + ' ===============')
        pl.trace('====================================================================================')

        # Get 30m CCDC & EVHR Masked Arrays
        bandPairIndices = bandPairIndicesList[bandPairIndex + 1]
        ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
        evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

        # Get 2m CCDC & EVHR Masked Arrays
        ccdcBandMaArrayRaw = iolib.fn_getma(fn_list[0], bandPairIndices[0])
        evhrBandMaArrayRaw = iolib.fn_getma(fn_list[1], bandPairIndices[1])

        ########################################
        # Mask threshold values (e.g., (median - threshold) < range < (median + threshold) 
        #  prior to generating common mask to reduce outliers ++++++[as per MC - 02/07/2022]
        ########################################
        evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])
        evhrBandMaArrayMedian = np.ma.median(evhrBandMaArray)
        pl.trace(' evhrBandMaArrayMedian median =' + str(np.ma.median(evhrBandMaArrayMedian)))    
        threshold = 500
        evhrBandMaArrayThresholdMin = evhrBandMaArrayMedian - threshold
        evhrBandMaArrayThresholdMax = evhrBandMaArrayMedian + threshold
        pl.trace(' evhrBandMaArrayThresholdMin = ' + str(evhrBandMaArrayThresholdMin))
        pl.trace(' evhrBandMaArrayThresholdMax = ' + str(evhrBandMaArrayThresholdMax))
        evhrBandMaThresholdMaxArray = np.ma.masked_where(evhrBandMaArray > evhrBandMaArrayThresholdMax, evhrBandMaArray) 
        evhrBandMaThresholdRangeArray = np.ma.masked_where(evhrBandMaThresholdMaxArray < evhrBandMaArrayThresholdMin, evhrBandMaThresholdMaxArray) 
        pl.trace(' evhrBandMaThresholdRangeArray median =' + str(np.ma.median(evhrBandMaThresholdRangeArray)))    
        evhrBandMaArray = evhrBandMaThresholdRangeArray

        # Generate common mask
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

        model_data_only_band = HuberRegressor().fit(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
    #    model_data_only_band = LinearRegression().fit(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
        pl.trace(str(bandNamePairList[bandPairIndex]) + '= > intercept: ' + str(
            model_data_only_band.intercept_) + ' slope: ' + str(model_data_only_band.coef_) + ' score: ' +
              str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)))
        pl.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                    model_data_only_band.intercept_, override=override)

        ########################################
        # #### Apply the model to the original EVHR (2m) to predict surface reflectance
        ########################################
        pl.trace(f'Applying model to {str(bandNamePairList[bandPairIndex])} in file {os.path.basename(fn_list[1])}')
        pl.trace(f'Input masked array shape: {evhrBandMaArray.shape}')

        score = model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band) 
        pl.trace(f'R2 score : {score}')
        sr_prediction_band = model_data_only_band.predict(evhrBandMaArrayRaw.ravel().reshape(-1, 1))
        pl.trace(f'Post-prediction shape : {sr_prediction_band.shape}')

        # Return to original shape and apply original mask
        orig_dims = evhrBandMaArrayRaw.shape
        evhr_sr_ma_band = np.ma.array(sr_prediction_band.reshape(orig_dims), mask=evhrBandMaArrayRaw.mask)

         # Check resulting ma
        pl.trace(f'Final masked array shape: {evhr_sr_ma_band.shape}')
    #    pl.trace('evhr_sr_ma=\n' + str(evhr_sr_ma_band))

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
    #     ccdc_evhr_srlite_list = [ccdc_warp_ma, evhr_sr_ma_band]
    #     compare_name_list = ['CCDC SR', 'EVHR SR-Lite']

    #     pl.plot_histograms(ccdc_evhr_srlite_list, fn_list, figsize=(5, 3),
    #                        title=str(bandNamePairList[bandPairIndex]) + " CCDC SR vs EVHR SR", override=override)
    #     pl.plot_maps(ccdc_evhr_srlite_list, compare_name_list, figsize=(10, 50), override=override)

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
        print(f"\nFinished with {str(bandNamePairList[bandPairIndex])} Band")
        
    return sr_prediction_list


# ## Set up inputs

# In[8]:


##############################################
# Default configuration values

start_time = time.time()  # record start time

# Debug levels:  0-no debug, 2-visualization, 3-detailed diagnostics
#debugLevel = 0
debugLevel = 0

# Toggle visualizations
# imagePlot = False
# histogramPlot = False
# scatterPlot = False
# fitPlot = False
imagePlot = True
histogramPlot = True
scatterPlot = True
fitPlot = True

override = False
pl = PlotLib(debugLevel, histogramPlot, scatterPlot, fitPlot)

# Temporary input - Need to pull from arg list
bandNamePairList = list([
    ['blue_ccdc', 'BAND-B'],
    ['green_ccdc', 'BAND-G'],
    ['red_ccdc', 'BAND-R'],
    ['nir_ccdc', 'BAND-N']])


# #### Look at the Projection (2 ways)

# In[9]:


def getProjection(r_fn, title):
    r_ds = iolib.fn_getds(r_fn)
    print(r_ds.GetProjection())
    if (debug_level >= 3): 
        print(r_ds.GetProjection())
        print("Driver: {}/{}".format(r_ds.GetDriver().ShortName,
                                r_ds.GetDriver().LongName))
        print("Size is {} x {} x {}".format(r_ds.RasterXSize,
                                        r_ds.RasterYSize,
                                        r_ds.RasterCount))
        print("Projection is {}".format(r_ds.GetProjection()))
        geotransform = r_ds.GetGeoTransform()
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

        # if (debug_level >= 2): 
        #     r_fn_data = xr.open_rasterio(r_ds, chunks=data_chunks)
        #     print(r_fn_data)
            
        pl.plot_combo(r_fn, figsize=(14,7), title=title)


# In[10]:


def validateBands(bandNamePairList, fn_list):
    ########################################
    # Validate Band Pairs and Retrieve Corresponding Array Indices
    ########################################
    pl.trace('bandNamePairList=' + str(bandNamePairList))
#    fn_full_list = [r_fn_ccdc, r_fn_evhr]
    bandPairIndicesList = getBandIndices(fn_list, bandNamePairList)
    pl.trace('bandIndices=' + str(bandPairIndicesList))
    return bandPairIndicesList


# In[11]:


def getIntersection(fn_list):
    # ########################################
    # # Align the CCDC and EVHR images, then take the intersection of the grid points
    # ########################################
    warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='average')
    warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
    return warp_ds_list, warp_ma_list


# In[12]:


def createImage(r_fn_evhr, numBandPairs, sr_prediction_list, name):

    ########################################
    # Create .tif image from band-based prediction layers
    ########################################
    pl.trace(f"\nApply coefficients to High Res File...{r_fn_evhr}")

    now = datetime.now()  # current date and time
    nowStr = now.strftime("%m%d%H%M")

    #  Derive file names for intermediate files
    head, tail = os.path.split(r_fn_evhr)
    filename = (tail.rsplit(".", 1)[0])
    output_name = "{}/{}".format(
        OUTDIR, name
#        nowStr, str(r_fn_evhr).split('/')[-1]
    ) + "_sr_02m.tif"
    pl.trace(f"\nCreating .tif image from band-based prediction layers...{output_name}")

    # Read metadata of EVHR file
    with rasterio.open(r_fn_evhr) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=numBandPairs-1)

    ########################################
    # Read each layer and write it to stack
    ########################################
    with rasterio.open(output_name, 'w', **meta) as dst:
        for id in range(1, numBandPairs):
            bandPrediction = sr_prediction_list[id]
            mean = bandPrediction.mean()
            pl.trace(f'BandPrediction.mean({mean})')
            dst.set_band_description(id, bandNamePairList[id - 1][1])
            bandPrediction1 = np.ma.masked_values(bandPrediction, -9999)
            dst.write_band(id, bandPrediction1)

    print("\nElapsed Time: " + output_name + ': ',
          (time.time() - start_time) / 60.0)  # time in min

    return output_name


# In[ ]:


from pathlib import Path
evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Yukon_Delta/5-toas"
#evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Senegal/5-toas"
#evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Fairbanks/5-toas"
#evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Siberia/5-toas"
ccdcdir = "/home/gtamkin/nobackup/dev/srlite/input/CCDC_v2"

outpath = OUTDIR = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/02272022/Yukon_Delta"
#outpath = OUTDIR = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/02272022/Senegal"
#outpath = OUTDIR = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/02272022/Fairbanks"
#outpath = OUTDIR = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/02272022/Siberia"

for r_fn_evhr in Path(evhrdir).glob("*.tif"):
    prefix = str(r_fn_evhr).rsplit("/", 1)
    name = str(prefix[1]).split("-toa.tif", 1)
    r_fn_ccdc = os.path.join(ccdcdir + '/' + name[0] + '-ccdc.tif')
    print ('\n Processing files: ', r_fn_evhr, r_fn_ccdc)

    fn_list = [str(r_fn_ccdc), str(r_fn_evhr)]

    pl.trace('\nCCDC file=' + str(r_fn_ccdc))
    getProjection(str(r_fn_ccdc), title="CCDC Combo Plot")
    
    pl.trace('\nEVHR file=' + str(r_fn_evhr))
    getProjection(str(r_fn_evhr), title="EVHR Combo Plot")
    
    bandPairIndicesList = validateBands(bandNamePairList, fn_list)
    
    warp_ds_list, warp_ma_list = getIntersection(fn_list)
    ccdc_warp_ma = warp_ma_list[0]
    evhr_warp_ma = warp_ma_list[1]
#    ccdc_warp_ma, evhr_warp_ma = warp_ma_list

    pl.trace('\n CCDC shape=' + str(warp_ma_list[0].shape) + ' EVHR shape=' + str(warp_ma_list[1].shape))

    pl.trace('\n Process Bands ....')
    sr_prediction_list = processBands(warp_ds_list, bandPairIndicesList, fn_list)

    pl.trace('\n Create Image....')
    outputname = createImage(str(r_fn_evhr), len(bandPairIndicesList), sr_prediction_list, name[0])

    # Use gdal_edit (via ILAB core SystemCommand) to convert GEE CCDC output to proper projection ESRI:102001 and set NoData value in place
    cogname = outputname.replace(".tif", "_cog.tif")
    command = 'gdalwarp -of cog ' + outputname + ' ' + cogname
    SystemCommand(command)

    break;

print("\nTotal Elapsed Time for: " + evhrdir + '/*.tif: ',
          (time.time() - start_time) / 60.0)  # time in min


# In[ ]:




