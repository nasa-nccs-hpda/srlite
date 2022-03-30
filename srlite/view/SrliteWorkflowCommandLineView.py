"""
Purpose: Build and apply regression model for coefficient identification of raster data using
         low resolution data (~30m) and CCDC inputs. Apply the coefficients to high resolution data (~2m)
         to generate a surface reflectance product (aka, SR-Lite). Usage requirements are referenced in README.

Data Source: This script has been tested with very high-resolution WV data.
             Additional testing will be required to validate the applicability
             of this model for other datasets.

Original Author: Glenn Tamkin, CISTO, Code 602
Portions Inspired by: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
import sys
import os
import ast
from datetime import datetime  # tracking date
import time  # tracking time
import argparse  # system libraries
import numpy as np
import rasterio
from osgeo import gdal
from srlite.model.PlotLib import PlotLib
from srlite.model.Context import Context
from pygeotools.lib import iolib, malib, geolib, filtlib, warplib
import osgeo
from osgeo import gdal
from core.model.SystemCommand import SystemCommand
from pathlib import Path

########################################
# Point to local pygeotools (not in ilab-kernel by default)
########################################
sys.path.append('/home/gtamkin/.local/lib/python3.9/site-packages')
#sys.path.append('/att/nobackup/gtamkin/dev/srlite/src')

# SR-Lite dependencies

# --------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------



def getBandIndices(fn_list, bandNamePair, pl):
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

def getProjection(r_fn, title, pl):
    r_ds = iolib.fn_getds(r_fn)
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

    pl.plot_combo(r_fn, figsize=(14, 7), title=title)

def validateBands(bandNamePairList, fn_list, pl):
    ########################################
    # Validate Band Pairs and Retrieve Corresponding Array Indices
    ########################################
    pl.trace('bandNamePairList=' + str(bandNamePairList))
    #    fn_full_list = [r_fn_ccdc, r_fn_evhr]
    bandPairIndicesList = getBandIndices(fn_list, bandNamePairList, pl)
    pl.trace('bandIndices=' + str(bandPairIndicesList))
    return bandPairIndicesList

def getIntersection(fn_list):
    # ########################################
    # # Align the CCDC and EVHR images, then take the intersection of the grid points
    # ########################################
    warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='average')
    warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
    return warp_ds_list, warp_ma_list

def createImage(r_fn_evhr, numBandPairs, sr_prediction_list, name,
                bandNamePairList, outdir, pl):
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
        outdir, name
        #        nowStr, str(r_fn_evhr).split('/')[-1]
    ) + "_sr_02m-precog.tif"
    pl.trace(f"\nCreating .tif image from band-based prediction layers...{output_name}")

    # Read metadata of EVHR file
    with rasterio.open(r_fn_evhr) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=numBandPairs - 1)

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

    return output_name

def getProjSrs(in_raster):
    from osgeo import gdal,osr
    ds=gdal.Open(in_raster)
    prj=ds.GetProjection()
    print(prj)

    srs=osr.SpatialReference(wkt=prj)
    print('srs=', srs)
    if srs.IsProjected:
        print (srs.GetAttrValue('projcs'))
    print (srs.GetAttrValue('geogcs'))
    return prj, srs

def getExtents(in_raster):
    from osgeo import gdal

    data = gdal.Open(in_raster, gdal.GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    extent = [minx, miny, maxx, maxy]
    data = None
    return extent

def getMetadata(band_num, input_file):
    src_ds = gdal.Open(input_file)
    if src_ds is None:
        print('Unable to open %s' % input_file)
        sys.exit(1)

    try:
        srcband = src_ds.GetRasterBand(band_num)
    except BaseException as err:
        print('No band %i found' % band_num)
        print(err)
        sys.exit(1)

    print("[ METADATA] = ", src_ds.GetMetadata())

    stats = srcband.GetStatistics(True, True)
    print("[ STATS ] = Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f", stats[0], stats[1], stats[2], stats[3])

    # source_layer = srcband.GetLayer()
    # x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # print ("[ EXTENTS] = ", x_min, x_max, y_min, y_max )

    print("[ NO DATA VALUE ] = ", srcband.GetNoDataValue())
    print("[ MIN ] = ", srcband.GetMinimum())
    print("[ MAX ] = ", srcband.GetMaximum())
    print("[ SCALE ] = ", srcband.GetScale())
    print("[ UNIT TYPE ] = ", srcband.GetUnitType())
    ctable = srcband.GetColorTable()

    if ctable is None:
        print('No ColorTable found')
        # sys.exit(1)
    else:
        print("[ COLOR TABLE COUNT ] = ", ctable.GetCount())
        for i in range(0, ctable.GetCount()):
            entry = ctable.GetColorEntry(i)
            if not entry:
                continue
            print("[ COLOR ENTRY RGB ] = ", ctable.GetColorEntryAsRGB(i, entry))

    outputType = gdal.GetDataTypeName(srcband.DataType)

    print(outputType)

    return srcband.DataType


def warp(in_raster, outraster, dstSRS, outputType, xRes, yRes, extent):
    from osgeo import gdal
    ds = gdal.Warp(in_raster, outraster,
                   dstSRS=dstSRS, outputType=outputType, xRes=yRes, yRes=yRes, outputBounds=extent)
    ds = None
    return outraster

def downscale(targetAttributesFile, inFile, outFile, xRes=30.0, yRes=30.0):
    import os.path
    if not os.path.exists(str(outFile)):
        outputType = getMetadata(1, str(targetAttributesFile))
        new_projection, new_srs = getProjSrs(targetAttributesFile)
        extent = getExtents(targetAttributesFile)
        print(extent)
        outFile = warp(outFile, inFile, dstSRS=new_srs, outputType=outputType, xRes=xRes, yRes=yRes, extent=extent)
        ds = None
    return outFile

def applyThreshold(min, max, bandMaArray, pl):
    ########################################
    # Mask threshold values (e.g., (median - threshold) < range < (median + threshold)
    #  prior to generating common mask to reduce outliers ++++++[as per MC - 02/07/2022]
    ########################################
    pl.trace('======== Applying threshold algorithm to first EVHR Band (Assume Blue) ========================')
    bandMaThresholdMaxArray = np.ma.masked_where(bandMaArray > max, bandMaArray)
    bandMaThresholdRangeArray = np.ma.masked_where(bandMaThresholdMaxArray < min, bandMaThresholdMaxArray)
    pl.trace(' threshold range median =' + str(np.ma.median(bandMaThresholdRangeArray)))
    return bandMaThresholdRangeArray

def processBands(warp_ds_list, bandNamePairList, bandPairIndicesList, fn_list, r_fn_cloudmask_warp, override, pl):

    import numpy.ma as ma

    ccdc_warp_ds = warp_ds_list[0]
    evhr_warp_ds = warp_ds_list[1]

    ########################################
    # ### PREPARE CLOUDMASK
    # After retrieving the masked array from the warped cloudmask, further reduce it by suppressing the one ("1") value pixels
    ########################################
    from sklearn.linear_model import HuberRegressor
    pl.trace('bandPairIndicesList: ' + str(bandPairIndicesList))
    numBandPairs = len(bandPairIndicesList)
    warp_ma_masked_band_series = [numBandPairs]
    sr_prediction_list = [numBandPairs]
    # debug_level = 3

    #  Get Masked array from warped Cloudmask - assumes only 1 band in mask to be applied to all
    cloudmaskWarpExternalBandMaArray = iolib.fn_getma(r_fn_cloudmask_warp, 1)
    pl.trace(f'\nBefore Mask -> cloudmaskWarpExternalBandMaArray')
    pl.trace(f'cloudmaskWarpExternalBandMaArray hist: {np.histogram(cloudmaskWarpExternalBandMaArray)}')
    pl.trace(f'cloudmaskWarpExternalBandMaArray shape: {cloudmaskWarpExternalBandMaArray.shape}')
    count_non_masked = ma.count(cloudmaskWarpExternalBandMaArray)
    count_masked = ma.count_masked(cloudmaskWarpExternalBandMaArray)
    pl.trace(f'cloudmaskWarpExternalBandMaArray ma.count (masked)=' + str(count_non_masked))
    pl.trace(f'cloudmaskWarpExternalBandMaArray ma.count_masked (non-masked)=' + str(count_masked))
    pl.trace(
        f'cloudmaskWarpExternalBandMaArray total count (masked + non-masked)=' + str(count_masked + count_non_masked))
    pl.trace(f'cloudmaskWarpExternalBandMaArray max=' + str(cloudmaskWarpExternalBandMaArray.max()))
    pl.plot_combo(cloudmaskWarpExternalBandMaArray, figsize=(14, 7), title='cloudmaskWarpExternalBandMaArray')

    # Create a mask where the pixel values equal to 'one' are suppressed because these correspond to clouds
    pl.trace(f'\nAfter Mask == 1.0 (sum should be 0 since all ones are masked -> cloudmaskWarpExternalBandMaArray')
    cloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(cloudmaskWarpExternalBandMaArray == 1.0,
                                                                cloudmaskWarpExternalBandMaArray)
    pl.trace(f'cloudmaskWarpExternalBandMaArrayMasked hist: {np.histogram(cloudmaskWarpExternalBandMaArrayMasked)}')
    pl.trace(f'cloudmaskWarpExternalBandMaArrayMasked shape: {cloudmaskWarpExternalBandMaArrayMasked.shape}')
    count_non_masked = ma.count(cloudmaskWarpExternalBandMaArrayMasked)
    count_masked = ma.count_masked(cloudmaskWarpExternalBandMaArrayMasked)
    pl.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count (masked)=' + str(count_non_masked))
    pl.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count_masked (non-masked)=' + str(count_masked))
    pl.trace(f'cloudmaskWarpExternalBandMaArrayMasked total count (masked + non-masked)=' + str(
        count_masked + count_non_masked))
    pl.trace(f'cloudmaskWarpExternalBandMaArrayMasked max=' + str(cloudmaskWarpExternalBandMaArrayMasked.max()))
    pl.plot_combo(cloudmaskWarpExternalBandMaArrayMasked, figsize=(14, 7),
                  title='cloudmaskWarpExternalBandMaArrayMasked')

    #         # Get Cloud mask (assumes 1 band per scene)
    #     cloudmaskArray = iolib.fn_getma(r_fn_cloudmask, 1)
    # #    cloudmaskArray = iolib.ds_getma(warp_ds_list[2], 1)
    #     pl.trace(f'cloudmaskArray array shape: {cloudmaskArray.shape}')
    #     cloudmaskMaArray = np.ma.masked_where(cloudmaskArray == 1, cloudmaskArray)
    #     pl.trace(f'cloudmaskMaArray array shape: {cloudmaskMaArray.shape}')

    minWarning = 0
    firstBand = True
    threshold = False
    if (threshold == True):
        # Apply range of -100 to 200 for Blue Band pixel mask and apply to each band
        evhrBandMaArrayThresholdMin = -100
        #    evhrBandMaArrayThresholdMax = 10000
        evhrBandMaArrayThresholdMax = 2000
        pl.trace(' evhrBandMaArrayThresholdMin = ' + str(evhrBandMaArrayThresholdMin))
        pl.trace(' evhrBandMaArrayThresholdMax = ' + str(evhrBandMaArrayThresholdMax))
        minWarning = evhrBandMaArrayThresholdMin

    ########################################
    # ### FOR EACH BAND PAIR,
    # now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
    ########################################
    for bandPairIndex in range(0, numBandPairs - 1):

        pl.trace('=>')
        pl.trace('====================================================================================')
        pl.trace('============== Start Processing Band #' + str(bandPairIndex + 1) + ' ===============')
        pl.trace('====================================================================================')

        # Retrieve band pair
        bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

        # Get 30m CCDC Masked Arrays
        # ccdcBandMaArray = iolib.ds_getma(ccdc_warp_ds, 1)
        # evhrBandMaArray = iolib.ds_getma(evhr_warp_ds, 2)
        ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
        evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

        #  Create single mask for all bands based on Blue-band threshold values
        #  Assumes Blue-band is first indice pair, so collect mask on 1st iteration only.
        if (threshold == True):
            if (firstBand == True):
                evhrBandMaArray = applyThreshold(evhrBandMaArrayThresholdMin, evhrBandMaArrayThresholdMax,
                                                 evhrBandMaArray)
                firstBand = False

        #  Create a common mask that intersects the CCDC, EVHR, and Cloudmask - this will then be used to correct the input EVHR & CCDC
        warp_ma_band_list_all = [ccdcBandMaArray, evhrBandMaArray, cloudmaskWarpExternalBandMaArrayMasked]
        common_mask_band_all = malib.common_mask(warp_ma_band_list_all)
        # pl.trace(f'common_mask_band_all hist: {np.histogram(common_mask_band_all)}')
        # pl.trace(f'common_mask_band_all array shape: {common_mask_band_all.shape}')
        # pl.trace(f'common_mask_band_all array sum: {common_mask_band_all.sum()}')
        # # pl.trace(f'common_mask_band_data_only array count: {common_mask_band_all.count}')
        # pl.trace(f'common_mask_band_all array max: {common_mask_band_all.max()}')
        # pl.trace(f'common_mask_band_all array min: {common_mask_band_all.min()}')
        # # plot_combo(common_mask_band_all, figsize=(14,7), title='common_mask_band_all')
        # count_non_masked = ma.count(int(common_mask_band_all))
        # count_masked = ma.count_masked(common_mask_band_all)
        # pl.trace(f'common_mask_band_all ma.count (masked)=' + str(count_non_masked))
        # pl.trace(f'common_mask_band_all ma.count_masked (non-masked)=' + str(count_masked))
        # pl.trace(f'common_mask_band_all total count (masked + non-masked)=' + str(count_masked + count_non_masked))

        # Apply the 3-way common mask to the CCDC and EVHR bands
        warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band_all),
                                    np.ma.array(evhrBandMaArray, mask=common_mask_band_all)]

        # Check the mins of each ma - they should be greater than 0
        for j, ma in enumerate(warp_ma_masked_band_list):
            j = j + 1
            if (ma.min() < minWarning):
                pl.trace("Warning: Masked array values should be larger than " + str(minWarning))
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
        #        model_data_only_band = LinearRegression().fit(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
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

        # Get 2m EVHR Masked Arrays
        evhrBandMaArrayRaw = iolib.fn_getma(fn_list[1], bandPairIndices[1])
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
                           title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR DELTA ",
                           override=override)
        pl.plot_maps([evhr_pre_post_ma_list[1],
                      evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]],
                     [compare_name_list[1],
                      str(bandNamePairList[bandPairIndex]) + ' Difference: TOA-SR-Lite'], (10, 50),
                     cmap_list=['RdYlGn', 'RdBu'], override=override)
        print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

    return sr_prediction_list

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():

    print("Command line executed: ", sys.argv)  # saving command into log file

    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time

    # --------------------------------------------------------------------------------
    # 0. Prepare for run - set log file for script if requested (-l command line option)
    # --------------------------------------------------------------------------------
    context = Context().getDict()
    print('Initializing SRLite Regression script with the following parameters')
    print(f'TOA Directory:    {context[Context.DIR_TOA]}')
    print(f'CCDC Directory:    {context[Context.DIR_CCDC]}')
    print(f'Cloudmask Directory:    {context[Context.DIR_CLOUDMASK]}')
    print(f'Output Directory: {context[Context.DIR_OUTPUT]}')
    print(f'Band pairs:    {context[Context.LIST_BAND_PAIRS]}')
    print(f'Regression Model:    {context[Context.REGRESSION_MODEL]}')
    print(f'Log: {context[Context.LOG_FLAG]}')

    # create output dir
    os.system(f'mkdir -p {context[Context.DIR_OUTPUT]}')

    # Debug levels:  0-no debug, 2-visualization, 3-detailed diagnostics
    debug_level = int(context[Context.DEBUG_LEVEL])

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
    regressionType = 'sklearn'

    pl = PlotLib(debug_level, histogramPlot, scatterPlot, fitPlot)
    bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))


    if (debug_level >= 2):
        print(sys.path)
        print(osgeo.gdal.VersionInfo())

    # Retrieve paths
    evhrdir = context[Context.DIR_TOA]
    ccdcdir = context[Context.DIR_CCDC]
    cloudmaskdir = cloudmaskWarpdir = context[Context.DIR_CLOUDMASK]
    outpath = context[Context.DIR_OUTPUT]

    # for r_fn_evhr in sorted(Path(evhrdir).glob("*.tif")):
    for r_fn_evhr in (Path(evhrdir).glob("*.tif")):
         prefix = str(r_fn_evhr).rsplit("/", 1)
         name = str(prefix[1]).split("-toa.tif", 1)
         r_fn_ccdc = os.path.join(ccdcdir + '/' + name[0] + '-ccdc.tif')
         r_fn_cloudmask = os.path.join(cloudmaskdir + '/' + name[0] + '-toa_pred.tif')
         r_fn_cloudmaskWarp = os.path.join(cloudmaskWarpdir + '/' + name[0] + '-toa_pred_warp.tif')

         print('\n Processing files: ', r_fn_evhr, r_fn_ccdc, r_fn_cloudmask)

         # Get attributes of raw EVHR tif and create plot - assumes same root name suffixed by "-toa.tif")
         pl.trace('\nEVHR file=' + str(r_fn_evhr))
         getProjection(str(r_fn_evhr), "EVHR Combo Plot", pl)

         # Get attributes of raw CCDC tif and create plot - assumes same root name suffixed by '-ccdc.tif')
         pl.trace('\nCCDC file=' + str(r_fn_ccdc))
         getProjection(str(r_fn_ccdc), "CCDC Combo Plot", pl)

         # Get attributes of raw cloudmask tif and create plot - assumes same root name suffixed by '-toa_pred.tif')
         pl.trace('\nCloudmask file=' + str(r_fn_cloudmask))
         getProjection(str(r_fn_cloudmask), "Cloudmask Combo Plot", pl)

         #  Warp cloudmask to attributes of EVHR - suffix root name with '-toa_pred_warp.tif')
         pl.trace('\nCloudmask Warp=' + str(r_fn_cloudmaskWarp))
         downscale(str(r_fn_evhr), str(r_fn_cloudmask), str(r_fn_cloudmaskWarp), xRes=30.0, yRes=30.0)
         getProjection(str(r_fn_cloudmaskWarp), "Cloudmask Warp Combo Plot", pl)
         #    break;

         # Validate that input band name pairs exist in EVHR & CCDC files
         fn_list = [str(r_fn_ccdc), str(r_fn_evhr)]
         bandPairIndicesList = validateBands(bandNamePairList, fn_list, pl)

         # Get the common pixel intersection values of the EVHR & CCDC files
         warp_ds_list, warp_ma_list = getIntersection(fn_list)
         #    ccdc_warp_ma = warp_ma_list[0]
         #    evhr_warp_ma = warp_ma_list[1]

         pl.trace('\n CCDC shape=' + str(warp_ma_list[0].shape) + ' EVHR shape=' + str(warp_ma_list[1].shape))

         pl.trace('\n Process Bands ....')
         sr_prediction_list = processBands(warp_ds_list, bandNamePairList, bandPairIndicesList, fn_list, r_fn_cloudmaskWarp, override, pl)

         pl.trace('\n Create Image....')
         outputname = createImage(str(r_fn_evhr), len(bandPairIndicesList), sr_prediction_list, name[0],
                                  bandNamePairList, outpath, pl)

         # Use gdalwarp to create Cloud-optimized Geotiff (COG)
         cogname = outputname.replace("-precog.tif", ".tif")
         command = 'gdalwarp -of cog ' + outputname + ' ' + cogname
         SystemCommand(command)
         if os.path.exists(outputname):
             os.remove(outputname)

         break;

    print("\nTotal Elapsed Time for " + cogname + ': ',
           (time.time() - start_time) / 60.0)  # time in min

if __name__ == "__main__":
    main()
