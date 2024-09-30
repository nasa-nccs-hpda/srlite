#!/usr/bin/env python
# coding: utf-8
import ast
import os
import os.path
import sys
from datetime import datetime

import numpy as np
import osgeo
import pandas as pd
import rasterio
import sklearn
from osgeo import gdal, osr
from pygeotools.lib import iolib, warplib, malib
from pylr2 import regress2
from sklearn.linear_model import HuberRegressor, LinearRegression
from pathlib import Path
from srlite.model.Context import Context
import numpy.ma as ma

import multiprocessing as multiprocessing
import re
import time 

# -----------------------------------------------------------------------------
# class RasterLib
#
# This class contains the business logic for the SR-Lite application.  It combines
# general-purpose raster utilities (built on top of numpy, pandas, pygeotools, and
# sklearn) with SR-Lite specific methods.
# -----------------------------------------------------------------------------
class RasterLib(object):
    DIR_TOA = 'toa'

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, debug_level, plot_lib):

        # Initialize serializable context for orchestration
        self._debug_level = debug_level
        self._plot_lib = plot_lib

        try:
            if (self._debug_level >= 1):
                self._plot_lib.trace(f'GDAL version: {osgeo.gdal.VersionInfo()}')
        except BaseException as err:
            print('ERROR - check gdal version: ', err)
            sys.exit(1)
        return

    # -------------------------------------------------------------------------
    # _validateParms()
    #
    # Verify that required parameters exist in Context()
    # -------------------------------------------------------------------------
    def _validateParms(self, context, requiredList):
        for parm in requiredList:
            parmExists = False
            for item in context:
                if (item == parm):
                    parmExists = True
                    break;
            if (parmExists == False):
                print("Error: Missing required parameter: " + str(parm))
                exit(1)

    # -------------------------------------------------------------------------
    # _representsInt()
    #
    # Verify that the string safely represents and integer
    # -------------------------------------------------------------------------
    def _representsInt(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    # -------------------------------------------------------------------------
    # b_getma()
    #
    # Get masked array from input GDAL Band
    # -------------------------------------------------------------------------
    def b_getma(self, b):
        b_ndv = iolib.get_ndv_b(b)
        #This is more appropriate for float, handles precision issues
        bma = np.ma.masked_values(b.ReadAsArray(), b_ndv, shrink=False)
        return bma

    # -------------------------------------------------------------------------
    # ds_getma()
    #
    # Given input dataset, return a masked array for the input band
    # -------------------------------------------------------------------------
    def ds_getma(self, ds, bnum=1):
        b = ds.GetRasterBand(bnum)
        return self.b_getma(b)

    # -------------------------------------------------------------------------
    # ma2_1d
    #
    # Given a masked array, convert to 1D and return unmasked array contents
    # -------------------------------------------------------------------------
    def ma2_1d(self, ma):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False]
        return np.array(unmasked)

    # -------------------------------------------------------------------------
    # ma2df()
    #
    # Given a masked array, convert it to a data frame
    # -------------------------------------------------------------------------
    def ma2df(self, ma, product, band):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False] 
        df = pd.DataFrame(unmasked)
        df.columns = [product + band]
        df[product + band] = df[product + band] * 0.0001

        # srlite-GI#25_Address_Maximum_TIFF_file_size_exceeded - reduce memory usage
        return (df).astype('float32')

    # -------------------------------------------------------------------------
    # getBandIndices()
    #
    # Validate band name pairs and return corresponding gdal indices
    # -------------------------------------------------------------------------
    def getBandIndices(self, context):
        self._validateParms(context, [Context.LIST_BAND_PAIRS, Context.FN_LIST])

        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))

        fn_list = context[Context.FN_LIST]
        ccdcDs = gdal.Open(fn_list[context[Context.LIST_INDEX_TARGET]], gdal.GA_ReadOnly)
        ccdcBands = ccdcDs.RasterCount
        evhrDs = gdal.Open(fn_list[context[Context.LIST_INDEX_TOA]], gdal.GA_ReadOnly)
        evhrBands = evhrDs.RasterCount

        numBandPairs = len(bandNamePairList)
        bandIndices = [numBandPairs]
        toaBandNames = []
        targetBandNames = []

        for bandPairIndex in range(0, numBandPairs):

            ccdcBandIndex = evhrBandIndex = -1
            currentBandPair = bandNamePairList[bandPairIndex]

            for ccdcIndex in range(1, ccdcBands + 1):
                # read in bands from image
                band = ccdcDs.GetRasterBand(ccdcIndex)
                bandDescription = band.GetDescription()
                bandName = currentBandPair[context[Context.LIST_INDEX_TARGET]]
                if (self._representsInt(bandName)):
                    ccdcBandIndex = int(bandName)
                    break
                if len(bandDescription) == 0:
                    ccdcBandIndex = bandPairIndex + 1
                    self._plot_lib.trace(
                        f"Band has no description {bandName} - assume index of current band  {ccdcBandIndex}")
                    break
                else:
                    if (bandDescription == bandName):
                        ccdcBandIndex = ccdcIndex
                        break

            for evhrIndex in range(1, evhrBands + 1):
                # read in bands from image
                band = evhrDs.GetRasterBand(evhrIndex)
                bandDescription = band.GetDescription()
                bandName = currentBandPair[context[Context.LIST_INDEX_TOA]]
                if (self._representsInt(bandName)):
                    evhrBandIndex = int(bandName)
                    break
                else:
                    if len(bandDescription) == 0:
                        evhrBandIndex = bandPairIndex + 1
                        self._plot_lib.trace(
                            f"Band has no description {bandName} - assume index of current band  {evhrBandIndex}")
                        break
                    else:
                        if (bandDescription == bandName):
                            evhrBandIndex = evhrIndex
                            break

            if ((ccdcBandIndex == -1) or (evhrBandIndex == -1)):
                ccdcDs = evhrDs = None
                self._plot_lib.trace(f"Invalid band pairs - verify correct name and case {currentBandPair}")
                exit(f"Invalid band pairs - verify correct name and case {currentBandPair}")

            bandIndices.append([ccdcBandIndex, evhrBandIndex])
            toaBandNames.append(currentBandPair[1])
            targetBandNames.append(currentBandPair[0])

        context[Context.LIST_TOA_BANDS] = toaBandNames
        context[Context.LIST_TARGET_BANDS] = targetBandNames

        ccdcDs = evhrDs = None
        self._plot_lib.trace(f'Band Names: {str(bandNamePairList)} Indices: {str(bandIndices)}')
        return bandIndices

    # -------------------------------------------------------------------------
    # getAttributeSnapshot()
    #
    # Get snapshot of attributes of EVHR, CCDC, and Cloudmask tifs and create plot
    # -------------------------------------------------------------------------
    def getAttributeSnapshot(self, context):
        self._validateParms(context, [Context.FN_TOA, Context.FN_TARGET, Context.FN_CLOUDMASK])

        self.getAttributes(str(context[Context.FN_TOA]), "EVHR Combo Plot")
        self.getAttributes(str(context[Context.FN_TARGET]), "CCDC Combo Plot")
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            self.getAttributes(str(context[Context.FN_CLOUDMASK]), "Cloudmask Combo Plot")

    # -------------------------------------------------------------------------
    # displayComboPlot()
    #
    # Display combo plot
    # -------------------------------------------------------------------------
    def displayComboPlot(self, r_fn, title=None):
        if (self._debug_level >= 2):
            self._plot_lib.plot_combo(r_fn, figsize=(14, 7), title=title)

    # -------------------------------------------------------------------------
    # getAttributes()
    #
    # Trace image geotransform diagnostics
    # -------------------------------------------------------------------------
    def getAttributes(self, r_fn, title=None):
        geotransform = None
        r_ds = iolib.fn_getds(r_fn)
        if (self._debug_level >= 1):
            self._plot_lib.trace("\nFile Name is {}".format(r_fn))
            self._plot_lib.trace("Raster Count is: {},  Size is: ({} x {})".format(
                r_ds.RasterCount, r_ds.RasterYSize, r_ds.RasterXSize))
            self._plot_lib.trace("Projection is {}".format(r_ds.GetProjection()))
            geotransform = r_ds.GetGeoTransform()
            if geotransform:
                self._plot_lib.trace(
                    f'Origin: ({geotransform[0]}, '
                    f'{geotransform[3]}), Resolution: ({geotransform[1]}, {geotransform[5]})  ')

        self.displayComboPlot(r_fn, title)

        r_ds = None
        return geotransform

    # -------------------------------------------------------------------------
    # setTargetAttributes()
    #
    # Trace image geotransform diagnostics
    # -------------------------------------------------------------------------
    def setTargetAttributes(self, context, r_fn):

        r_ds = iolib.fn_getds(r_fn)
        context[Context.TARGET_GEO_TRANSFORM] = r_ds.GetGeoTransform()
        context[Context.TARGET_DRIVER] = r_ds.GetDriver()
        context[Context.TARGET_PRJ] = r_ds.GetProjection()
        context[Context.TARGET_SRS] = r_ds.GetSpatialRef()
        context[Context.TARGET_RASTERX_SIZE] = r_ds.RasterXSize
        context[Context.TARGET_RASTERY_SIZE] = r_ds.RasterYSize
        context[Context.TARGET_RASTER_COUNT] = r_ds.RasterCount
        r_ds = None

    # -------------------------------------------------------------------------
    # replaceNdv()
    #
    # Replace no data value in gdal dataset
    # -------------------------------------------------------------------------
    def replaceNdv(self, context, src_fn, new_ndv):
        out_fn = os.path.splitext(src_fn)[0] + '_ndv.tif'
        self.removeFile(out_fn, context[Context.CLEAN_FLAG])
        if (not os.path.exists(out_fn)):
            ds = gdal.Open(src_fn)
            b = ds.GetRasterBand(1)
            # Extract old ndv
            old_ndv = iolib.get_ndv_b(b)

            # Load masked array
            bma = iolib.ds_getma(ds)

            # Handle cases with input ndv of nan
            # if old_ndv == np.nan:
            bma = np.ma.fix_invalid(bma)

            # Set new fill value
            bma.set_fill_value(new_ndv)
            # Fill ma with new value and write out
            iolib.writeGTiff(bma.filled(), out_fn, ds, ndv=new_ndv)
        return out_fn

    # -------------------------------------------------------------------------
    # getStatistics()
    #
    # Get snapshot of statistics of EVHR, CCDC, and Cloudmask tifs
    # -------------------------------------------------------------------------
    def getStatistics(self, context):
        self._validateParms(context, [Context.MA_LIST, Context.MA_WARP_LIST])

        # Get snapshot of attributes of EVHR, CCDC, and Cloudmask tifs and create plot")
        [warp_ma.get_fill_value() for warp_ma in context[Context.MA_WARP_LIST]]

        self._plot_lib.trace('\nAll these arrays should now have the same shape:\n')
        [self._plot_lib.trace(ma.shape) for ma in context[Context.MA_WARP_LIST]]
        self._plot_lib.trace('Input array mins/maxs')
        [self._plot_lib.trace(f'input ma min: {ma.min()}') for ma in context[Context.MA_LIST]]
        [self._plot_lib.trace(f'input ma max: {ma.max()}') for ma in context[Context.MA_LIST]]
        self._plot_lib.trace('Warped array mins/maxs')
        [self._plot_lib.trace(f'warped ma min: {ma.min()}') for ma in context[Context.MA_WARP_LIST]]
        [self._plot_lib.trace(f'warped ma max: {ma.max()}') for ma in context[Context.MA_WARP_LIST]]

    # -------------------------------------------------------------------------
    # alignNoDataValues()
    #
    # Align NoDataValues in image list to match target
    # -------------------------------------------------------------------------
    def alignNoDataValues(self, context):
        self._validateParms(context, [Context.FN_LIST, Context.FN_REPROJECTION_LIST, Context.TARGET_FN])

        # Ensure that all NoData values match TARGET_FN (e.g., TOA)
        # Retrieve No Data Value from TOA as guiding attribute for all inputs (as per MC)
        context[Context.TARGET_NODATA_VALUE] = dst_ndv = self.get_ndv(str(context[Context.FN_TOA]))
        for fn in context[Context.FN_REPROJECTION_LIST]:
            current_ndv = self.get_ndv(fn)
            if (current_ndv != context[Context.TARGET_NODATA_VALUE]):
                out_fn = self.replaceNdv(context, fn, context[Context.TARGET_NODATA_VALUE])
                index = context[Context.FN_LIST].index(fn)
                context[Context.FN_LIST][index] = out_fn
                index = context[Context.FN_REPROJECTION_LIST].index(fn)
                context[Context.FN_REPROJECTION_LIST][index] = out_fn
        return dst_ndv

    # -------------------------------------------------------------------------
    # getReprojection()
    #
    # Reproject inputs to TOA attributes (res, extent, srs, nodata) and return masked arrays of reprojected pixels
    # -------------------------------------------------------------------------
    def getReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST, Context.FN_REPROJECTION_LIST, Context.TARGET_FN,
                                      Context.TARGET_SAMPLING_METHOD])

        # Ensure equivalent NoDataValues
        dst_ndv = self.alignNoDataValues(context)

        # Reproject inputs to TOA attributes (res, extent, srs, nodata)
        src_fn = context[Context.FN_REPROJECTION_LIST]
        warp_ds_list = []
        for fn in src_fn:
            
            # Derive reprojected file name
            out_fn = os.path.splitext(fn)[0]
            out_fn = os.path.basename(out_fn)
            out_fn_warp = context[Context.DIR_OUTPUT_WARP] + '/' + out_fn + str(Context.FN_WARP_SUFFIX)

             # Remove existing SR-Lite output if clean_flag is activated
            self.removeFile(out_fn_warp, context[Context.CLEAN_FLAG])

            if (os.path.exists(out_fn_warp)):
                ds_warp = gdal.Open(out_fn_warp)  
                warp_ds_list.append(ds_warp)
            else:
                ds_warp = warplib.diskwarp_multi_fn([fn],
                                                res=context[Context.TARGET_XRES],
                                                extent=str(context[Context.TARGET_FN]),
                                                t_srs=str(context[Context.TARGET_FN]),
                                                r=context[Context.TARGET_SAMPLING_METHOD],
                                                dst_ndv=dst_ndv,
                                                outdir=context[Context.DIR_OUTPUT_WARP])
                warp_ds_list.append(ds_warp[0])

        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]

        return warp_ds_list, warp_ma_list

    # -------------------------------------------------------------------------
    # alignInputs()
    #
    # Align inputs to TOA attributes (res, extent, srs, nodata) and return masked arrays of reprojected pixels
    # -------------------------------------------------------------------------
    def alignInputs(self, toa, context):
        #  Reproject (downscale) TOA to CCDC resolution (30m)  - use 'average' for resampling method
        #  Reproject TARGET (CCDC) to remaining attributes of EVHR TOA Downscale (extent, srs, etc.) 
        context[Context.FN_REPROJECTION_LIST] = [str(context[Context.FN_TARGET]), str(toa)]
        context[Context.TARGET_FN] = str(toa)
        context[Context.TARGET_SAMPLING_METHOD] = 'average'
        context[Context.DS_WARP_LIST], context[Context.MA_WARP_LIST] = self.getReprojection(context)

        #  Reproject cloudmask to attributes of EVHR TOA Downscale  - use 'mode' for resampling method
        if eval(context[Context.CLOUD_MASK_FLAG]):
            context[Context.FN_LIST].append(str(context[Context.FN_CLOUDMASK]))
            context[Context.FN_REPROJECTION_LIST] = [str(context[Context.FN_CLOUDMASK])]
            context[Context.TARGET_FN] = str(toa)
                
            # Reproject to 'mode' sampling for regression
            context[Context.TARGET_SAMPLING_METHOD] = 'mode'
            context[Context.DS_WARP_CLOUD_LIST], context[
                Context.MA_WARP_CLOUD_LIST] = self.getReprojection(context)
                
            context[Context.LIST_INDEX_CLOUDMASK] = 2


    # -------------------------------------------------------------------------
    # prepareEVHRCloudmask()
    #
    # Mask out clouds.  We are expecting cloud pixel values equal to 0 or 1.
    # We are filtering values > 0.5 because it is faster than values == 1.0
    # -------------------------------------------------------------------------
    def prepareEVHRCloudmask(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_INDEX_CLOUDMASK])

        # Mask out clouds
        cloudmask_warp_ma = context[Context.MA_WARP_CLOUD_LIST][0]
        cloudmaskWarpExternalBandMaArrayMasked = \
            np.ma.masked_where(cloudmask_warp_ma >= 0.5, cloudmask_warp_ma)
#        np.ma.masked_where(cloudmask_warp_ma == 1.0, cloudmask_warp_ma)

        return cloudmaskWarpExternalBandMaArrayMasked

    # -------------------------------------------------------------------------
    # prepareQualityFlagMask()
    #
    # Suppress values=[0, 3, 4] according to Band #8 because they correspond to NoData,
    # Clouds and Cloud Shadows
    # -------------------------------------------------------------------------
    def prepareQualityFlagMask(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_INDEX_CLOUDMASK])

        # Mask out clouds
        cloudmask_warp_ds_target = context[Context.DS_WARP_LIST][context[Context.LIST_INDEX_TARGET]]
        cloudmask_warp_ma = iolib.ds_getma(cloudmask_warp_ds_target, 8)
        self._plot_lib.trace(
            f'\nSuppress values=[0, 3, 4] according to Band #8 because they correspond to NoData, Clouds, '
            f'and Cloud Shadows')

        ndv = int(Context.DEFAULT_NODATA_VALUE)
        cloudmaskWarpExternalBandArrayData = np.ma.getdata(cloudmask_warp_ma)
        cloudmaskWarpExternalBandArrayDataQfFiltered = np.select(
            [cloudmaskWarpExternalBandArrayData == 0, cloudmaskWarpExternalBandArrayData == 3,
             cloudmaskWarpExternalBandArrayData == 4], [ndv, ndv, ndv], cloudmaskWarpExternalBandArrayData)
        cloudmaskWarpExternalBandArrayDataQfNdvFiltered = np.select(
            [cloudmaskWarpExternalBandArrayDataQfFiltered != ndv], [0.0],
            cloudmaskWarpExternalBandArrayDataQfFiltered)
        cloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(
            cloudmaskWarpExternalBandArrayDataQfNdvFiltered == ndv,
            cloudmaskWarpExternalBandArrayDataQfNdvFiltered)
        return cloudmaskWarpExternalBandMaArrayMasked

    # -------------------------------------------------------------------------
    # prepareMasks()
    #
    # Prepare optional masks
    # -------------------------------------------------------------------------
    def prepareMasks(self, context):

        # Get optional Cloudmask
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            context['cloudmaskEVHRWarpExternalBandMaArrayMasked'] = self.prepareEVHRCloudmask(context)

        # Get optional Quality flag mask
        if (eval(context[Context.QUALITY_MASK_FLAG])):
            context['cloudmaskQFWarpExternalBandMaArrayMasked'] = self.prepareQualityFlagMask(context)

    # -------------------------------------------------------------------------
    # getCommonMask()
    #
    # Aggregate optional masks to create a common mask that intersects the CCDC/QF, EVHR, and Cloudmasks.
    # Mask out ALL negative values in input if requested.
    # -------------------------------------------------------------------------
    def getCommonMask(self, context, targetBandArray, toaBandArray):

        context['evhrBandMaThresholdArray'] = None
        warp_ma_band_list_all = [targetBandArray, toaBandArray]
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            warp_ma_band_list_all.append(context['cloudmaskEVHRWarpExternalBandMaArrayMasked'])
        if (eval(context[Context.QUALITY_MASK_FLAG])):
            warp_ma_band_list_all.append(context['cloudmaskQFWarpExternalBandMaArrayMasked'])
        if (eval(context[Context.THRESHOLD_MASK_FLAG])):
            if (context['evhrBandMaThresholdArray'] == None):
                #  Create single mask for all bands based on Blue-band threshold values
                #  Assumes Blue-band is first indice pair, so collect mask on 1st iteration only.
                context['evhrBandMaThresholdArray'] = self._applyThreshold(context[Context.THRESHOLD_MIN],
                                                                           context[Context.THRESHOLD_MAX],
                                                                           toaBandArray)
                firstBand = False
            warp_ma_band_list_all.append(context['evhrBandMaThresholdArray'])

        # Mask negative values in input (if requested)
        if (eval(context[Context.POSITIVE_MASK_FLAG])):
            warp_valid_ma_band_list_all = [np.ma.masked_where(ma < 0, ma) for ma in warp_ma_band_list_all]
        else:
            warp_valid_ma_band_list_all = warp_ma_band_list_all

        # Create common mask
        common_mask_band_all = malib.common_mask(warp_valid_ma_band_list_all)

        return common_mask_band_all

    # -------------------------------------------------------------------------
    # mean_bias_error()
    #
    # Derive deltas in mean values between observed and predicted arrays
    # -------------------------------------------------------------------------
    def mean_bias_error(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_true = y_true.reshape(len(y_true),1)
        y_pred = y_pred.reshape(len(y_pred),1)
        diff = (y_true-y_pred)
        mbe = diff.mean()
        return mbe

    # -------------------------------------------------------------------------
    # band_performance()
    #
    # Calculate band metrics based on corresponding TOA (i.e., EVHR) and REF (i.e., CCDC) arrays.
    # -------------------------------------------------------------------------
    def band_performance(self, metadata, target_sr_data_only_band, toa_sr_data_only_band):
        target_sr_data_only_band_reshaped = target_sr_data_only_band.values.reshape(-1,1)
        metadata['r2_score'] = sklearn.metrics.r2_score(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['explained_variance'] = sklearn.metrics.explained_variance_score(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['mbe'] = self.mean_bias_error(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['mae'] = sklearn.metrics.mean_absolute_error(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['mape'] = sklearn.metrics.mean_absolute_percentage_error(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['medae'] = sklearn.metrics.median_absolute_error(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['mse'] = sklearn.metrics.mean_squared_error(target_sr_data_only_band_reshaped, toa_sr_data_only_band)
        metadata['rmse'] = metadata['mse'] ** 0.5
        metadata['mean_ref_sr'] = target_sr_data_only_band.mean()
        metadata['mean_src_sr'] = toa_sr_data_only_band.mean()
        metadata['mae_norm'] = metadata['mae'] / metadata['mean_ref_sr']
        metadata['rmse_norm'] =  metadata['rmse'] / metadata['mean_ref_sr']
        return metadata

    # -------------------------------------------------------------------------
    # sr_performance()
    #
    # Generate band metrics based on corresponding TOA and CCDC arrays.
    # Optionally override metrics with no data value (used for synthetic bands)
    # -------------------------------------------------------------------------
    def sr_performance(self, df, context, bandName, model, intercept, slope, ndv_value=None):

        metadata = {}
        metadata['band_name'] = bandName
        metadata['model'] = model
        metadata['intercept'] = intercept
        metadata['slope'] = slope

        if (ndv_value == None):
            sr = df[df['Band'] == bandName]
            toa_sr_data_only_band = sr['EVHR_SRLite']
            target_sr_data_only_band = sr['CCDC_SR']
            metadata = self.band_performance(metadata, target_sr_data_only_band, toa_sr_data_only_band)
        else:
            metadata['r2_score'] = ndv_value
            metadata['explained_variance'] = ndv_value
            metadata['mbe'] = ndv_value
            metadata['mae'] = ndv_value
            metadata['mape'] = ndv_value
            metadata['medae'] = ndv_value
            metadata['mse'] = ndv_value
            metadata['rmse'] = ndv_value
            metadata['mean_ref_sr'] = ndv_value
            metadata['mean_src_sr'] = ndv_value
            metadata['mae_norm'] = ndv_value
            metadata['rmse_norm'] = ndv_value

        metadata['catid'] = context[Context.CAT_ID] 
        metadata['input_src_toa']  = context[Context.FN_TOA]
        metadata['input_ref_sr']  = context[Context.FN_TARGET]
        metadata['input_cloudmask']  = context[Context.FN_CLOUDMASK]

        return metadata

    # -------------------------------------------------------------------------
    # calculateStatistics()
    #
    # Calculate statistics for each band, assuming a specific band ordering.
    # First four bands are constant [B,G,R,N].  When 8Band processing in effect, four
    # synthetic bands are appended to the list [C,Y,RE,N2]
    # -------------------------------------------------------------------------
    def calculateStatistics(self, context):

        # Get coefficients for standard 4-Bands
        sr_metrics_list = context[Context.METRICS_LIST]
        model = context[Context.REGRESSION_MODEL]

        # Correction coefficients for simulated bands
        yellowGreenCorr = 0.473
        yellowRedCorr = 0.527
        rededgeRedCorr = 0.621
        rededgeNIR1Corr = 0.379

        try:
            # Retrieve slope, intercept, and score coefficients for data-driven bands
            blueSlope = sr_metrics_list['slope'][0]
            blueIntercept = sr_metrics_list['intercept'][0]

            greenSlope = sr_metrics_list['slope'][1]
            greenIntercept = sr_metrics_list['intercept'][1]

# TODO - Add check to see if redSlope exists (assume)
            redSlope = sr_metrics_list['slope'][2]
            redIntercept = sr_metrics_list['intercept'][2]

            NIR1Slope = sr_metrics_list['slope'][3]
            NIR1Intercept = sr_metrics_list['intercept'][3]

            # Reproject newly minted sr-lite output for status calculations
            evhrSrliteImage = os.path.join(context[Context.FN_COG])
            fn_list = [evhrSrliteImage]
            warp_sr_ds_list = warplib.memwarp_multi_fn(fn_list, 
                                                        res=30, 
                                                        extent=evhrSrliteImage,
                                                        t_srs=evhrSrliteImage,
                                                        r='average', 
                                                        dst_ndv=context[Context.TARGET_NODATA_VALUE])

            # Aggregate warped CCDC and Cloudmask from initialization with reprojected SR sandwiched between
            warp_ds_list = [context[Context.DS_WARP_LIST][0], warp_sr_ds_list[0], 
                            context[Context.DS_WARP_CLOUD_LIST][0]]
            
            # Generate masked arrays for calculations
            warp_ds_list_multiband = warp_ds_list[0:2]
            warp_ma_list_blu = [self.ds_getma(ds, 1) for ds in warp_ds_list_multiband]
            warp_ma_list_grn = [self.ds_getma(ds, 2) for ds in warp_ds_list_multiband]
            warp_ma_list_red = [self.ds_getma(ds, 3) for ds in warp_ds_list_multiband]
            warp_ma_list_nir = [self.ds_getma(ds, 4) for ds in warp_ds_list_multiband]
            cloud_warp_ma = iolib.ds_getma(warp_ds_list[2], 1).astype(np.uint8)

            # divvy up into band-specific arrays across CCDC + SRLite according to input source
            ccdc_warp_ma_blu, evhrsr_warp_ma_blu = warp_ma_list_blu
            ccdc_warp_ma_grn, evhrsr_warp_ma_grn = warp_ma_list_grn
            ccdc_warp_ma_red, evhrsr_warp_ma_red = warp_ma_list_red
            ccdc_warp_ma_nir, evhrsr_warp_ma_nir = warp_ma_list_nir

            warp_ma_list = [ccdc_warp_ma_blu, ccdc_warp_ma_grn, ccdc_warp_ma_red, ccdc_warp_ma_nir,
                            evhrsr_warp_ma_blu, evhrsr_warp_ma_grn, evhrsr_warp_ma_red, evhrsr_warp_ma_nir,
                            cloud_warp_ma]

            # Create cloudmask according to threshold that Matt suggested
            cloudfree_warp_ma = np.ma.masked_where(cloud_warp_ma >= 0.5, cloud_warp_ma)
            warp_ma_list_cloudfree = warp_ma_list[0:8] + [cloudfree_warp_ma]
            common_mask = malib.common_mask(warp_ma_list_cloudfree)
            warp_ma_masked_list = [np.ma.array(ma, mask=common_mask) for ma in warp_ma_list_cloudfree]

            # Create a dataframe with the arrays
            reflect_df = pd.concat([
                self.ma2df(warp_ma_masked_list[0], 'CCDC_SR', 'Blue'),
                self.ma2df(warp_ma_masked_list[1], 'CCDC_SR', 'Green'),
                self.ma2df(warp_ma_masked_list[2], 'CCDC_SR', 'Red'),
                self.ma2df(warp_ma_masked_list[3], 'CCDC_SR', 'NIR'),
                self.ma2df(warp_ma_masked_list[4], 'EVHR_SRLite', 'Blue'),
                self.ma2df(warp_ma_masked_list[5], 'EVHR_SRLite', 'Green'),
                self.ma2df(warp_ma_masked_list[6], 'EVHR_SRLite', 'Red'),
                self.ma2df(warp_ma_masked_list[7], 'EVHR_SRLite', 'NIR')],
                axis=1)

            # Convert wide table to long table
            reflectanceTypes = ['CCDC_SR','EVHR_SRLite']
            reflect_df_long = pd.wide_to_long(reflect_df.reset_index(),
                                            stubnames=reflectanceTypes,
                                            i='index', j='Band', suffix='\D+') \
                .reset_index()

            from pandas.api.types import CategoricalDtype
            bandsType = CategoricalDtype(categories = ['Blue','Green','Red','NIR'], ordered=True)
            reflect_df_long['Band'] = reflect_df_long['Band'].astype(bandsType)

            # Create table of dataframes with coefficients & statistics
            metrics_srlite_long = None
            if eval(context[Context.BAND8_FLAG]):
                ndv_value = "NA"
                metrics_srlite_long = pd.concat([
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Blue', model, blueIntercept, blueSlope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Green', model, greenIntercept, greenSlope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Red', model, redIntercept, redSlope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'NIR', model, NIR1Intercept, NIR1Slope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Coastal', model, blueIntercept, blueSlope, ndv_value)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Yellow', model,
                                                    (greenIntercept * yellowGreenCorr) + (redIntercept * yellowRedCorr),
                                                    (greenSlope * yellowGreenCorr) + (redSlope * yellowRedCorr),
                                                    ndv_value)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'RedEdge', model,
                                                    (redIntercept * rededgeRedCorr) + (NIR1Intercept * rededgeNIR1Corr),
                                                    (redSlope * rededgeRedCorr) + (NIR1Slope * rededgeNIR1Corr),
                                                    ndv_value)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'NIR2', model, NIR1Intercept, NIR1Slope, ndv_value)])
                ]).reset_index()
            else:
                metrics_srlite_long = pd.concat([
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Blue', model, blueIntercept, blueSlope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Green', model, greenIntercept, greenSlope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'Red', model, redIntercept, redSlope)]),
                    pd.DataFrame([self.sr_performance(reflect_df_long, context,
                                                    'NIR', model, NIR1Intercept, NIR1Slope)])
                ]).reset_index()

        except BaseException as err:
            issue = "CSV file creation failed, likely due to conflicts with the following expected band ordering: " \
                "First four bands must be constant [B,G,R,N].  If 8-Band processing is requested, four synthetic bands are " \
                    "then appended to the list [C,Y,RE,N2]: Error details = " + str(err)

            raise Exception(issue)
        return context[Context.FN_COG], metrics_srlite_long

    # -------------------------------------------------------------------------
    # generateCSV()
    #
    # Write out statistics per band to a csv file
    # -------------------------------------------------------------------------

    def generateErrorReport(self, context):

       if (eval(str(context[Context.ERROR_REPORT_FLAG]))):
 
            if not (len(context[Context.ERROR_LIST]) == 0):
                df = pd.DataFrame(context[Context.ERROR_LIST])
                
                path = os.path.join(context[Context.DIR_OUTPUT_ERROR],
                                    Context.DEFAULT_ERROR_REPORT_SUFFIX)
                
                # Remove existing error report if clean_flag is activated
                self.removeFile(path, context[Context.CLEAN_FLAG])

                df.to_csv(path)
                self._plot_lib.trace(
                    f"\n\nGenerated error report: {path}")
            else:
               self._plot_lib.trace("\nNo errors reported.")

    # -------------------------------------------------------------------------
    # generateCSV()
    #
    # Write out statistics per band to a csv file
    # -------------------------------------------------------------------------
    def generateCSV(self, context):

        # Generate simulated band statistics
        context[Context.FN_COG_8BAND], context[Context.METRICS_LIST] = \
                self.calculateStatistics(context)

        if (eval(context[Context.CSV_FLAG])):
            if (context[Context.BATCH_NAME] != 'None'):
                figureBase = context[Context.BATCH_NAME] + '_' + context[Context.FN_PREFIX] 
            else:
                figureBase = context[Context.FN_PREFIX] 

            path = os.path.join(context[Context.DIR_OUTPUT_CSV],
                                figureBase + Context.DEFAULT_STATISTICS_REPORT_SUFFIX)
            context[Context.METRICS_LIST].drop('index', axis=1, inplace=True)
            context[Context.METRICS_LIST].to_csv(path)
            self._plot_lib.trace(
                f"\nCreated CSV with coefficients for batch {context[Context.BATCH_NAME]}...\n   {path}")


    # -------------------------------------------------------------------------
    # predictSurfaceReflectance()
    #
    # Perform regression fit based on model type (TARGET against TOA)
    # -------------------------------------------------------------------------
    def predictSurfaceReflectance(self, context, band_name, toaBandMaArrayRaw,
                                  target_warp_ma_masked_band, toa_warp_ma_masked_band, sr_metrics_list):

        try:
            # Perform regression fit based on model type (TARGET against TOA)
            target_sr_band = target_warp_ma_masked_band.ravel()
            toa_sr_band = toa_warp_ma_masked_band.ravel()
            sr_prediction_band_2m = None
            model_data_only_band = None
            metadata = {}

            target_sr_data_only_band = target_sr_band[target_sr_band.mask == False]
            target_sr_data_only_band_reshaped = target_sr_data_only_band.reshape(-1, 1)
            toa_sr_data_only_band = toa_sr_band[toa_sr_band.mask == False]
            toa_sr_data_only_band_reshaped = toa_sr_data_only_band.reshape(-1, 1)

            ####################
            ### Huber (robust) Regressor
            ####################
            if (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_HUBER):
                # ravel the Y band (e.g., CCDC)
                # - /home/gtamkin/.conda/envs/ilab_gt/lib/python3.7/site-packages/sklearn/utils/validation.py:993:
                # DataConversion Warning: A column-vector y was passed when a 1d array was expected.
                # Please change the shape of y to (n_samples, ), for example using ravel().
                model_data_only_band = HuberRegressor().fit(
                    toa_sr_data_only_band_reshaped, target_sr_data_only_band_reshaped.ravel())

                sr_prediction_band_2m = model_data_only_band.predict(toaBandMaArrayRaw.reshape(-1, 1))

                #  band-specific metadata
                metadata = self._model_coeffs_(context,
                                                band_name,
                                                toaBandMaArrayRaw,
                                                model_data_only_band.intercept_,
                                                model_data_only_band.coef_[0])

            ####################
            ### OLS (simple) Regressor
            ####################
            elif (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_OLS):
                model_data_only_band = LinearRegression().fit(
                    toa_sr_data_only_band_reshaped, target_sr_data_only_band_reshaped)

                sr_prediction_band_2m = model_data_only_band.predict(toaBandMaArrayRaw.reshape(-1, 1))

                #  band-specific metadata
                metadata = self._model_coeffs_(context,
                                                band_name,
                                                toaBandMaArrayRaw,
                                                model_data_only_band.intercept_,
                                                model_data_only_band.coef_[0])

            ####################
            ### Reduced Major Axis (rma) Regressor
            ####################
            elif (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_RMA):

                reflect_df = pd.concat([
                    self.ma2df(toa_sr_data_only_band, 'EVHR_TOA', 'Band'), 
                    self.ma2df(target_sr_data_only_band, 'CCDC_SR', 'Band')],
                    axis=1)
                model_data_only_band = regress2(np.array(reflect_df['EVHR_TOABand']), np.array(reflect_df['CCDC_SRBand']),
                                                _method_type_2="reduced major axis")

                #  band-specific metadata
                metadata = self._model_coeffs_(context,
                                            band_name,
                                            toaBandMaArrayRaw,
                                            model_data_only_band['intercept'],
                                            model_data_only_band['slope'])

            # Calculate SR-Lite band using original TOA 2m band
                sr_prediction_band_2m = self.calculate_prediction_band(context,
                                                                band_name,
                                                                metadata,
                                                                sr_metrics_list)
            else:
                print('Invalid regressor specified %s' % context[Context.REGRESSION_MODEL])
                sys.exit(1)

            self._plot_lib.trace(f"\nRegressor=[{context[Context.REGRESSION_MODEL]}] "
            f"slope={metadata['slope']} intercept={metadata['intercept']} ]")

        except BaseException as err:
                print('\npredictSurfaceReflectance processing failed - Error details: ', err)
                raise err
        
        return sr_prediction_band_2m, metadata

    # -------------------------------------------------------------------------
    # predictSurfaceReflectance()
    #
    # Perform regression fit based on model type (TARGET against TOA)
    # -------------------------------------------------------------------------
    def predictSurfaceReflectanceConcurrent(self, reg_model, band_name, toaBandMaArrayRaw,
                                  target_warp_ma_masked_band, toa_warp_ma_masked_band, sr_metrics_list):

        try:
            # Perform regression fit based on model type (TARGET against TOA)
            target_sr_band = target_warp_ma_masked_band.ravel()
            toa_sr_band = toa_warp_ma_masked_band.ravel()
            sr_prediction_band_2m = None
            model_data_only_band = None
            metadata = {}

            target_sr_data_only_band = target_sr_band[target_sr_band.mask == False]
            target_sr_data_only_band_reshaped = target_sr_data_only_band.reshape(-1, 1)
            toa_sr_data_only_band = toa_sr_band[toa_sr_band.mask == False]
            toa_sr_data_only_band_reshaped = toa_sr_data_only_band.reshape(-1, 1)

            ####################
            ### Huber (robust) Regressor
            ####################
            if (reg_model == Context.REGRESSOR_MODEL_HUBER):
                # ravel the Y band (e.g., CCDC)
                # - /home/gtamkin/.conda/envs/ilab_gt/lib/python3.7/site-packages/sklearn/utils/validation.py:993:
                # DataConversion Warning: A column-vector y was passed when a 1d array was expected.
                # Please change the shape of y to (n_samples, ), for example using ravel().
                model_data_only_band = HuberRegressor().fit(
                    toa_sr_data_only_band_reshaped, target_sr_data_only_band_reshaped.ravel())

                sr_prediction_band_2m = model_data_only_band.predict(toaBandMaArrayRaw.reshape(-1, 1))

                #  band-specific metadata
                metadata = self._model_coeffs_(reg_model,
                                                band_name,
                                                toaBandMaArrayRaw,
                                                model_data_only_band.intercept_,
                                                model_data_only_band.coef_[0])

            ####################
            ### OLS (simple) Regressor
            ####################
            elif (reg_model == Context.REGRESSOR_MODEL_OLS):
                model_data_only_band = LinearRegression().fit(
                    toa_sr_data_only_band_reshaped, target_sr_data_only_band_reshaped)

                sr_prediction_band_2m = model_data_only_band.predict(toaBandMaArrayRaw.reshape(-1, 1))

                #  band-specific metadata
                metadata = self._model_coeffs_(reg_model,
                                                band_name,
                                                toaBandMaArrayRaw,
                                                model_data_only_band.intercept_,
                                                model_data_only_band.coef_[0])

            ####################
            ### Reduced Major Axis (rma) Regressor
            ####################
            elif (reg_model == Context.REGRESSOR_MODEL_RMA):

                reflect_df = pd.concat([
                    self.ma2df(toa_sr_data_only_band, 'EVHR_TOA', 'Band'),
                    self.ma2df(target_sr_data_only_band, 'CCDC_SR', 'Band')],
                    axis=1)
                model_data_only_band = regress2(np.array(reflect_df['EVHR_TOABand']), np.array(reflect_df['CCDC_SRBand']),
                                                _method_type_2="reduced major axis")

                #  band-specific metadata
                metadata = self._model_coeffs_Concurrent(reg_model,
                                            band_name,
                                            toaBandMaArrayRaw,
                                            model_data_only_band['intercept'],
                                            model_data_only_band['slope'])

            # Calculate SR-Lite band using original TOA 2m band
                sr_prediction_band_2m = self.calculate_prediction_band_Concurrent(
                                                                band_name,
                                                                metadata,
                                                                sr_metrics_list)
            else:
                print('Invalid regressor specified %s' % reg_model)
                sys.exit(1)

            self._plot_lib.trace(f"\nRegressor=[{reg_model}] "
            f"slope={metadata['slope']} intercept={metadata['intercept']} ]")

        except BaseException as err:
                print('\npredictSurfaceReflectanceConcurrent processing failed - Error details: ', err)
                raise err
        
        return sr_prediction_band_2m, metadata
    # -------------------------------------------------------------------------
    # calculate_prediction_band()
    #
    # Perform regression fit based on model type (TARGET against TOA)
    # -------------------------------------------------------------------------
    def calculate_prediction_band_Concurrent(self,
                                  band_name, metadata, sr_metrics_list):
        sr_prediction_band_2m = self.calculate_prediction_band(None, band_name, metadata, sr_metrics_list)
        return sr_prediction_band_2m
 
    def calculate_prediction_band(self, context,
                                  band_name, metadata, sr_metrics_list):
        try:
            sr_prediction_band_2m = None
            toaBandMaArrayRaw = metadata['toaBandMaArrayRaw']
            slope = metadata['slope']
            intercept = metadata['intercept']

            # Correction coefficients for simulated bands
            yellowGreenCorr = 0.473
            yellowRedCorr = 0.527
            rededgeRedCorr = 0.621
            rededgeNIR1Corr = 0.379

            if (band_name == 'BAND-C'):
                # Apply BAND-B coefficients to Coastal band since we have no corresponding CCDC (as per Matt)
                slope = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-B','slope'].values[0]
                intercept = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-B','intercept'].values[0]
                sr_prediction_band_2m = (toaBandMaArrayRaw.astype(float) * slope) + (intercept * 10000)

            elif (band_name == 'BAND-Y'):
                # Apply BAND-G and BAND-R coefficients to RedEdge band since we have no corresponding CCDC (as per Matt)
                green_slope  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-G','slope'].values[0]
                green_intercept  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-G','intercept'].values[0]
                red_slope  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-R','slope'].values[0]
                red_intercept  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-R','intercept'].values[0]

                correctedGreenBand = ((toaBandMaArrayRaw.astype(float) * green_slope) + (green_intercept * 10000)) * yellowGreenCorr
                correctedRedBand = ((toaBandMaArrayRaw.astype(float) * red_slope) + (red_intercept * 10000)) * yellowRedCorr

                sr_prediction_band_2m = correctedGreenBand + correctedRedBand

            elif (band_name == 'BAND-RE'):
                # Apply BAND-R and BAND-N coefficients to RedEdge band since we have no corresponding CCDC (as per Matt)
                red_slope  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-R','slope'].values[0]
                red_intercept  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-R','intercept'].values[0]
                nir1_slope  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-N','slope'].values[0]
                nir1_intercept  = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-N','intercept'].values[0]

                correctedRedBand = ((toaBandMaArrayRaw.astype(float) * red_slope) + (red_intercept * 10000)) * rededgeRedCorr
                correctedNirBand = ((toaBandMaArrayRaw.astype(float) * nir1_slope) + (nir1_intercept * 10000)) * rededgeNIR1Corr

                sr_prediction_band_2m = correctedRedBand + correctedNirBand

            elif (band_name == 'BAND-N2'):
                # Apply BAND-N coefficients to NIR2 band since we have no corresponding CCDC (as per Matt)
                slope = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-N','slope'].values[0]
                intercept = sr_metrics_list.loc[sr_metrics_list.band_name=='BAND-N','intercept'].values[0]
                sr_prediction_band_2m = (toaBandMaArrayRaw.astype(float)  * slope) + (intercept * 10000)

            else:
                # Calculate SR-Lite band using original TOA 2m band
                sr_prediction_band_2m = (toaBandMaArrayRaw.astype(float)  * slope) + (intercept * 10000)

        except BaseException as err:
                print('\ncalculate_prediction_band processing failed - Error details: ', err)
                raise err

        # srlite-GI#25_Address_Maximum_TIFF_file_size_exceeded - reduce memory usage
        return sr_prediction_band_2m.astype('float32')

    # -------------------------------------------------------------------------
    # _model_coeffs_()
    #
    # Populate dictionary of coefficients
    # -------------------------------------------------------------------------
    def _model_coeffs_(self, context, band_name, toaBandMaArrayRaw, intercept, slope):

        metadata = {}
        metadata['band_name'] = band_name
        metadata['model'] = context[Context.REGRESSION_MODEL]
        metadata['intercept'] = intercept
        metadata['slope'] = slope
        metadata['toaBandMaArrayRaw'] = toaBandMaArrayRaw

        return metadata

    # -------------------------------------------------------------------------
    # _model_coeffs_()
    #
    # Populate dictionary of coefficients
    # -------------------------------------------------------------------------

    def _model_coeffs_Concurrent(self, reg_model, band_name, toaBandMaArrayRaw, intercept, slope):

        metadata = {}
        metadata['band_name'] = band_name
        metadata['model'] = reg_model
        metadata['intercept'] = intercept
        metadata['slope'] = slope
        metadata['toaBandMaArrayRaw'] = toaBandMaArrayRaw

        return metadata

    # -------------------------------------------------------------------------
    # processBandPairIndex()
    #
    # Populate dictionary of coefficients
    # -------------------------------------------------------------------------
    def processBandPairIndexPathos(self, context, bandPairIndicesList, bandPairIndex, warp_ds_list, 
                       bandNamePairList, common_mask_list, minWarning, sr_unmasked_prediction_list, sr_prediction_list, sr_metrics_list):
            
        try:
            self._plot_lib.trace('=>')
            self._plot_lib.trace('====================================================================================')
            self._plot_lib.trace('== Start Processing Band #' + str(bandPairIndex + 1) + ' ' + 
                                 str(bandPairIndicesList[bandPairIndex + 1]) + ' ===============')
            self._plot_lib.trace('====================================================================================')

            # Retrieve band pair
            bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

            # Get 30m EVHR/CCDC Masked Arrays
            targetBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
            toaBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

            # Create common mask based on user-specified list (e.g., cloudmask, threshold, QF)
            context[Context.COMMON_MASK] = self.getCommonMask(context, targetBandMaArray, toaBandMaArray)
            common_mask_list.append(context[Context.COMMON_MASK])

            # Apply the 3-way common mask to the CCDC and EVHR bands
            warp_ma_masked_band_list = [np.ma.array(targetBandMaArray, mask=context[Context.COMMON_MASK]),
                                        np.ma.array(toaBandMaArray, mask=context[Context.COMMON_MASK])]

            # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
            #                    exit(1)

            ########################################
            # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
            # CCDC SR is first element in list, which needs to be the y-var:
            # b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
            ########################################

            # Get 2m TOA Masked Array
            toaIndexArray = bandPairIndicesList[bandPairIndex+1]
            toaIndex = toaIndexArray[1]
            toaBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA], toaIndex)
            sr_prediction_band, metadata = self.predictSurfaceReflectance(context,
                                                                          bandNamePairList[bandPairIndex][1],
                                                                          toaBandMaArrayRaw,
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TARGET]],
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TOA]],
                                                                          sr_metrics_list)

            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file '
                f'{os.path.basename(context[Context.FN_LIST][context[Context.LIST_INDEX_TOA]])}')
            self._plot_lib.trace(f'Metrics: {metadata}')

            ########### save predictions for each band #############
            sr_unmasked_prediction_list.append(sr_prediction_band)

            # Return to original shape and apply original mask
            toa_sr_ma_band_reshaped = sr_prediction_band.reshape(toaBandMaArrayRaw.shape)

            toa_sr_ma_band = np.ma.array(
                toa_sr_ma_band_reshaped,
                mask=toaBandMaArrayRaw.mask)
            sr_prediction_list.append(toa_sr_ma_band)

            #  ########### save metadata for each band #############
            # if (bandPairIndex == 0):
            #     context[Context.METRICS_LIST] = pd.concat([pd.DataFrame([metadata], index=[bandPairIndex])])
            # else:
            #     context[Context.METRICS_LIST] = pd.concat([context[Context.METRICS_LIST], pd.DataFrame([metadata], index=[bandPairIndex])])

#            ########### save metadata for each band #############
            if len(context[Context.METRICS_LIST]) == 0:
                # if str(Context.METRICS_LIST) not in context.keys():
#            if (bandPairIndex == 0):
                context[Context.METRICS_LIST] = pd.concat([pd.DataFrame([metadata], index=[0])])
            else:
                rows = len(context[Context.METRICS_LIST].index)
                context[Context.METRICS_LIST] = pd.concat([context[Context.METRICS_LIST], pd.DataFrame([metadata], index=[rows])])

            print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

            context['currentBandPairIndex'] = bandPairIndex

        except BaseException as err:
                print('\nprocessBandPairIndexPathos processing failed - Error details: ', err)
                # ########### save error for each failed TOA #############
                # metadata = {}
                # metadata['toa_name'] = str(toa)
                # metadata['error'] = str(err)
                # if (errorIndex == 0):
                #     sr_errors_list = pd.concat([pd.DataFrame([metadata], index=[errorIndex])])
                # else:
                #     sr_errors_list = pd.concat([sr_errors_list, pd.DataFrame([metadata], index=[errorIndex])])
                # errorIndex = errorIndex + 1
                raise err
   
        return context

    # -------------------------------------------------------------------------
    # processBandPairIndex()
    #
    # Populate dictionary of coefficients
    # -------------------------------------------------------------------------
    def processBandPairIndex(self, context, bandPairIndicesList, bandPairIndex, warp_ds_list, 
                       bandNamePairList, common_mask_list, minWarning, sr_unmasked_prediction_list, sr_prediction_list):
            
            self._plot_lib.trace('=>')
            self._plot_lib.trace('====================================================================================')
            self._plot_lib.trace('== Start Processing Band #' + str(bandPairIndex + 1) + ' ' + 
                                 str(bandPairIndicesList[bandPairIndex + 1]) + ' ===============')
            self._plot_lib.trace('====================================================================================')

            # Retrieve band pair
            bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

            # Get 30m EVHR/CCDC Masked Arrays
            targetBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
            toaBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

            # Create common mask based on user-specified list (e.g., cloudmask, threshold, QF)
            context[Context.COMMON_MASK] = self.getCommonMask(context, targetBandMaArray, toaBandMaArray)
            common_mask_list.append(context[Context.COMMON_MASK])

            # Apply the 3-way common mask to the CCDC and EVHR bands
            warp_ma_masked_band_list = [np.ma.array(targetBandMaArray, mask=context[Context.COMMON_MASK]),
                                        np.ma.array(toaBandMaArray, mask=context[Context.COMMON_MASK])]

            # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
            #                    exit(1)

            ########################################
            # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
            # CCDC SR is first element in list, which needs to be the y-var:
            # b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
            ########################################

            # Get 2m TOA Masked Array
            toaIndexArray = bandPairIndicesList[bandPairIndex+1]
            toaIndex = toaIndexArray[1]
            toaBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA], toaIndex)
            sr_prediction_band, metadata = self.predictSurfaceReflectance(context,
                                                                          bandNamePairList[bandPairIndex][1],
                                                                          toaBandMaArrayRaw,
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TARGET]],
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TOA]],
                                                                          sr_metrics_list)

            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file '
                f'{os.path.basename(context[Context.FN_LIST][context[Context.LIST_INDEX_TOA]])}')
            self._plot_lib.trace(f'Metrics: {metadata}')

            ########### save predictions for each band #############
            sr_unmasked_prediction_list.append(sr_prediction_band)

            # Return to original shape and apply original mask
            toa_sr_ma_band_reshaped = sr_prediction_band.reshape(toaBandMaArrayRaw.shape)

            toa_sr_ma_band = np.ma.array(
                toa_sr_ma_band_reshaped,
                mask=toaBandMaArrayRaw.mask)
            sr_prediction_list.append(toa_sr_ma_band)

            ########### save metadata for each band #############
            if (bandPairIndex == 0):
                sr_metrics_list = pd.concat([pd.DataFrame([metadata], index=[bandPairIndex])])
            else:
                sr_metrics_list = pd.concat([sr_metrics_list, pd.DataFrame([metadata], index=[bandPairIndex])])

            print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

            return sr_metrics_list
    
    # -------------------------------------------------------------------------
    # simulateSurfaceReflectance()
    #
    # Perform workflow to create simulated surface reflectance for each band (SR-Lite)
    # This method hosts the primary orchestration logic for the SR-Lite application.
    # -------------------------------------------------------------------------
    def _simulateSurfaceReflectance(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]

        sr_prediction_list = []
        sr_unmasked_prediction_list = []
        sr_metrics_list = []
        common_mask_list = []
        warp_ds_list = context[Context.DS_WARP_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        minWarning = 0

        # Aggregate the requested masks (e.g., clouds, quality mask)
        self.prepareMasks(context)

        ########################################
        # ### FOR EACH BAND PAIR,
        # now, each input should have same exact dimensions, grid, projection.
        # They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
        ########################################
        #for bandPairIndex in range(0, len(bandPairIndicesList) - 1):
        num_workers = len(bandPairIndicesList)   
        items = [(context, bandPairIndicesList, bandPairIndex, warp_ds_list, 
                bandNamePairList, common_mask_list, minWarning, 
                sr_unmasked_prediction_list, sr_prediction_list) for bandPairIndex in range(0, num_workers)]

        from multiprocessing.pool import Pool
        print('max processes: ', multiprocessing.cpu_count(), ' processes requested from pool: ', num_workers)
        print(f'Starting pool.starmap_async() for toas: {str(bandPairIndicesList)}', flush=True)
        with Pool(num_workers) as pool:
            # issue tasks to process pool
            result = pool.starmap_async(self.processBandPairIndex, items)
            # iterate results
            for result in result.get():
                print(f'Got result: {result}', flush=True)

        # remove transient TOA arrays
        sr_metrics_list.drop('toaBandMaArrayRaw', axis=1, inplace=True)
        # sr_metrics_list.drop('index', axis=1, inplace=True)
        sr_metrics_list.reset_index()

        return sr_prediction_list, sr_metrics_list, common_mask_list

    # -------------------------------------------------------------------------
    # simulateSurfaceReflectance()
    #
    # Perform workflow to create simulated surface reflectance for each band (SR-Lite)
    # This method hosts the primary orchestration logic for the SR-Lite application.
    # -------------------------------------------------------------------------
    def simulateSurfaceReflectancePathos(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        from pathos.multiprocessing import ProcessingPool,ThreadingPool
        tmap = ThreadingPool().map
        # amap = ProcessingPool().amap            
        
        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]

        sr_prediction_list = []
        sr_unmasked_prediction_list = []
        context[Context.METRICS_LIST] = []
        common_mask_list = []
        warp_ds_list = context[Context.DS_WARP_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        minWarning = 0

        # Aggregate the requested masks (e.g., clouds, quality mask)
        self.prepareMasks(context)

        ########################################
        # ### FOR EACH BAND PAIR,
        # now, each input should have same exact dimensions, grid, projection.
        # They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
        ########################################
        #for bandPairIndex in range(0, len(bandPairIndicesList) - 1):
        num_workers = len(bandPairIndicesList)   
        # items = [(context, bandPairIndicesList, bandPairIndex, warp_ds_list, 
        #         bandNamePairList, common_mask_list, minWarning, 
        #         sr_unmasked_prediction_list, sr_prediction_list) for bandPairIndex in range(0, num_workers)]

        # newContext = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [0,1,2], [warp_ds_list], 
        #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])


        # results = [num_workers]
        for i in range(num_workers-1):
            print(f'Starting ProcessingPool().tmap() for band pair: {str(bandPairIndicesList[i])}', flush=True)
            tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [i], [warp_ds_list], 
                       [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], 
                       [sr_prediction_list], [context[Context.METRICS_LIST]])
            time.sleep(10)
            print(f'End ProcessingPool().tmap() for band pair: {str(bandPairIndicesList[i])}', flush=True)
        #     results[i] = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [i], [warp_ds_list], 
        #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], 
        #                [sr_prediction_list], [sr_metrics_list])
        #     # results.append(result)

        # for j in range(num_workers):
        #     result = results[j].get()
        #     print(f'Ending ProcessingPool().amap() for toa: {str(bandPairIndicesList[j])} {result}', flush=True)

        # print(f'Ending ProcessingPool().tmap() for band pairs: {str(bandPairIndicesList)}', flush=True)
        # first = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [0], [warp_ds_list], 
        #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])
        # second = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [1], [warp_ds_list], 
        #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])

        # tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [0], [warp_ds_list], 
        #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])
        # tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [1], [warp_ds_list], 
        #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])

        # # print("second.get() = ", second.get())
        # # print("first.get() = ", first.get())
        # # for result in result.get():
        # # context = newContext[0]
        # print(f'last index processed: {second[0]["currentBandPairIndex"]}', flush=True)
 
        # from multiprocessing.pool import Pool
        # print('max processes: ', multiprocessing.cpu_count(), ' processes requested from pool: ', num_workers)
        # print(f'Starting pool.starmap_async() for toas: {str(bandPairIndicesList)}', flush=True)
        # with Pool(num_workers) as pool:
        #     # issue tasks to process pool
        #     result = pool.starmap_async(self.processBandPairIndex, items)
        #     # iterate results
        #     for result in result.get():
        #         print(f'Got result: {result}', flush=True)

        # remove transient TOA arrays
        context[Context.METRICS_LIST].drop('toaBandMaArrayRaw', axis=1, inplace=True)
        # sr_metrics_list.drop('index', axis=1, inplace=True)
        context[Context.METRICS_LIST].reset_index()

        return sr_prediction_list, context[Context.METRICS_LIST], common_mask_list

    # -------------------------------------------------------------------------
    # simulateSurfaceReflectance()
    #
    # Perform workflow to create simulated surface reflectance for each band (SR-Lite)
    # This method hosts the primary orchestration logic for the SR-Lite application.
    # -------------------------------------------------------------------------
    def simulateSurfaceReflectancePathosWorks(self, context):

        try:
            self._validateParms(context,
                                [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                                Context.REGRESSION_MODEL, Context.FN_LIST])

            from pathos.multiprocessing import ThreadingPool
            tmap = ThreadingPool().map
            
            bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]

            sr_prediction_list = []
            sr_unmasked_prediction_list = []
            sr_metrics_list = []
            common_mask_list = []
            warp_ds_list = context[Context.DS_WARP_LIST]
            bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
            minWarning = 0

            # Aggregate the requested masks (e.g., clouds, quality mask)
            self.prepareMasks(context)

            ########################################
            # ### FOR EACH BAND PAIR,
            # now, each input should have same exact dimensions, grid, projection.
            # They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
            ########################################
            #for bandPairIndex in range(0, len(bandPairIndicesList) - 1):
            num_workers = len(bandPairIndicesList)   
            items = [(context, bandPairIndicesList, bandPairIndex, warp_ds_list, 
                    bandNamePairList, common_mask_list, minWarning, 
                    sr_unmasked_prediction_list, sr_prediction_list) for bandPairIndex in range(0, num_workers)]

            # newContext = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [0,1,2], [warp_ds_list], 
            #                [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])


            print(f'Starting ProcessingPool().tmap() for toas: {str(bandPairIndicesList)}', flush=True)
            first = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [0], [warp_ds_list], 
                        [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])
            second = tmap(self.processBandPairIndexPathos, [context], [bandPairIndicesList], [1], [warp_ds_list], 
                        [bandNamePairList], [common_mask_list], [minWarning], [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])
            # print("second.get() = ", second.get())
            # print("first.get() = ", first.get())
            # for result in result.get():
            # context = newContext[0]
            print(f'last index processed: {second[0]["currentBandPairIndex"]}', flush=True)
    
            # from multiprocessing.pool import Pool
            # print('max processes: ', multiprocessing.cpu_count(), ' processes requested from pool: ', num_workers)
            # print(f'Starting pool.starmap_async() for toas: {str(bandPairIndicesList)}', flush=True)
            # with Pool(num_workers) as pool:
            #     # issue tasks to process pool
            #     result = pool.starmap_async(self.processBandPairIndex, items)
            #     # iterate results
            #     for result in result.get():
            #         print(f'Got result: {result}', flush=True)

            # remove transient TOA arrays
            context[Context.METRICS_LIST].drop('toaBandMaArrayRaw', axis=1, inplace=True)
            # sr_metrics_list.drop('index', axis=1, inplace=True)
            context[Context.METRICS_LIST].reset_index()

        except BaseException as err:
            print('\simulateSurfaceReflectancePathosWorks processing failed - Error details: ', err)
            raise err

        return sr_prediction_list, context[Context.METRICS_LIST], common_mask_list

    # -------------------------------------------------------------------------
    # simulateSurfaceReflectance()
    #
    # Perform workflow to create simulated surface reflectance for each band (SR-Lite)
    # This method hosts the primary orchestration logic for the SR-Lite application.
    # -------------------------------------------------------------------------
    def simulateSurfaceReflectance(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]

        sr_prediction_list = []
        sr_unmasked_prediction_list = []
        sr_metrics_list = []
        common_mask_list = []
        warp_ds_list = context[Context.DS_WARP_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        minWarning = 0

        # Aggregate the requested masks (e.g., clouds, quality mask)
        self.prepareMasks(context)

        ########################################
        # ### FOR EACH BAND PAIR,
        # now, each input should have same exact dimensions, grid, projection.
        # They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
        ########################################
        for bandPairIndex in range(0, len(bandPairIndicesList) - 1):

            self._plot_lib.trace('=>')
            self._plot_lib.trace('====================================================================================')
            self._plot_lib.trace('============== Start Processing Band #' + str(bandPairIndex + 1) + ' ===============')
            self._plot_lib.trace('====================================================================================')

            # Retrieve band pair
            bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

            # Get 30m EVHR/CCDC Masked Arrays
            targetBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
            toaBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

            # Create common mask based on user-specified list (e.g., cloudmask, threshold, QF)
            context[Context.COMMON_MASK] = self.getCommonMask(context, targetBandMaArray, toaBandMaArray)
            common_mask_list.append(context[Context.COMMON_MASK])

            # Apply the 3-way common mask to the CCDC and EVHR bands
            warp_ma_masked_band_list = [np.ma.array(targetBandMaArray, mask=context[Context.COMMON_MASK]),
                                        np.ma.array(toaBandMaArray, mask=context[Context.COMMON_MASK])]

            # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
            #                    exit(1)

            ########################################
            # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
            # CCDC SR is first element in list, which needs to be the y-var:
            # b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
            ########################################

            # Get 2m TOA Masked Array
            toaIndexArray = bandPairIndicesList[bandPairIndex+1]
            toaIndex = toaIndexArray[1]
            toaBandMaArrayRaw = iolib.fn_getma(str(context[Context.FN_TOA]), toaIndex)
            sr_prediction_band, metadata = self.predictSurfaceReflectance(context,
                                                                          bandNamePairList[bandPairIndex][1],
                                                                          toaBandMaArrayRaw,
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TARGET]],
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TOA]],
                                                                          sr_metrics_list)

            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file '
                f'{os.path.basename(context[Context.FN_LIST][context[Context.LIST_INDEX_TOA]])}')
            self._plot_lib.trace(f'Metrics: {metadata}')

            ########### save predictions for each band #############
            sr_unmasked_prediction_list.append(sr_prediction_band)

            # Return to original shape and apply original mask
            toa_sr_ma_band_reshaped = sr_prediction_band.reshape(toaBandMaArrayRaw.shape)

            toa_sr_ma_band = np.ma.array(
                toa_sr_ma_band_reshaped,
                mask=toaBandMaArrayRaw.mask)
            sr_prediction_list.append(toa_sr_ma_band)

            ########### save metadata for each band #############
            if (bandPairIndex == 0):
                sr_metrics_list = pd.concat([pd.DataFrame([metadata], index=[bandPairIndex])])
            else:
                sr_metrics_list = pd.concat([sr_metrics_list, pd.DataFrame([metadata], index=[bandPairIndex])])

            print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

        # remove transient TOA arrays
        sr_metrics_list.drop('toaBandMaArrayRaw', axis=1, inplace=True)
        sr_metrics_list.reset_index()

        return sr_prediction_list, sr_metrics_list, common_mask_list

   # -------------------------------------------------------------------------
    # createImage()
    #
    # Convert list of prediction arrays to TIF image.  Optionally create a
    # Cloud-Optimized GeoTIF (COG).
    # -------------------------------------------------------------------------
    def _createImage(self, context):
        self._validateParms(context, [Context.DIR_OUTPUT, Context.FN_PREFIX,
                                      Context.CLEAN_FLAG,
                                      Context.FN_SRC,
                                      Context.FN_DEST,
                                      Context.LIST_BAND_PAIRS, Context.PRED_LIST,
                                      Context.LIST_TOA_BANDS])

        ########################################
        # Create .tif image from band-based prediction layers
        ########################################
        self._plot_lib.trace(f"\nApply coefficients to "
                             f"{context[Context.BAND_NUM]}-Band High Res File...\n   "
                             f"{str(context[Context.FN_SRC])}")

        now = datetime.now()  # current date and time

        context[Context.FN_SUFFIX] = str(Context.FN_SRLITE_NONCOG_SUFFIX)

        #  Derive file names for intermediate files
        output_name = "{}/{}".format(
            context[Context.DIR_OUTPUT], str(context[Context.FN_PREFIX])
        ) + str(context[Context.FN_SUFFIX])

        # Remove pre-COG image
        fileExists = (os.path.exists(output_name))
        if fileExists and (eval(context[Context.CLEAN_FLAG])):
            self.removeFile(output_name, context[Context.CLEAN_FLAG])

        # Read metadata of EVHR file
        with rasterio.open(str(context[Context.FN_SRC])) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        numBandPairs = int(context[Context.BAND_NUM])
        meta.update(count=numBandPairs)

        meta.update({
            "nodata": context[Context.TARGET_NODATA_VALUE],
            "descriptions": context[Context.BAND_DESCRIPTION_LIST]
        })

        band_data_list = context[Context.PRED_LIST]
        band_description_list = list(context[Context.BAND_DESCRIPTION_LIST])

        ########################################
        # Read each layer and write it to stack
        ########################################
        with rasterio.open(output_name, 'w', **meta) as dst:
            for id in range(0, numBandPairs):
                bandPrediction = band_data_list[id]
                dst.set_band_description(id + 1, str(band_description_list[id]))
                bandPrediction1 = np.ma.masked_values(bandPrediction, context[Context.TARGET_NODATA_VALUE])
                dst.write_band(id + 1, bandPrediction1)

        if (not (eval(context[Context.NONCOG_FLAG]))):
            # Create Cloud-optimized Geotiff (COG)
            context[Context.FN_SRC] = str(output_name)
            context[Context.FN_DEST] = "{}/{}".format(
                context[Context.DIR_OUTPUT], str(context[Context.FN_PREFIX])
            ) + str(Context.FN_SRLITE_SUFFIX)
            output_name = self.createCOG(context)
            self._plot_lib.trace(f"\nCreated COG from stack of regressed bands...\n   {output_name}")
            
        return output_name

  # -------------------------------------------------------------------------
    # createImage()
    #
    # Convert list of prediction arrays to TIF image.  Optionally create a
    # Cloud-Optimized GeoTIF (COG).
    # -------------------------------------------------------------------------
    def createImage(self, context):
        self._validateParms(context, [Context.DIR_OUTPUT, Context.FN_PREFIX,
                                      Context.CLEAN_FLAG,
                                      Context.FN_SRC,
                                      Context.FN_DEST,
                                      Context.LIST_BAND_PAIRS, Context.PRED_LIST,
                                      Context.LIST_TOA_BANDS])

        ########################################
        # Create .tif image from band-based prediction layers
        ########################################
        self._plot_lib.trace(f"\nApply coefficients to "
                             f"{context[Context.BAND_NUM]}-Band High Res File...\n   "
                             f"{str(context[Context.FN_SRC])}")
        
        #  Derive file names for intermediate files
        intermediate_output_name = "{}/{}".format(
            context[Context.DIR_OUTPUT], str(context[Context.FN_PREFIX])
        ) + str(Context.FN_SRLITE_NONCOG_SUFFIX)

        # Define pointers to datasets
        ds_toa = None
        ds_toa_copy_GTiff = None

        try:

            # Remove pre-COG image
            fileExists = (os.path.exists(intermediate_output_name))
            if fileExists and (eval(context[Context.CLEAN_FLAG])):
                self.removeFile(intermediate_output_name, context[Context.CLEAN_FLAG])

            # Get metadata
            numBandPairs = int(context[Context.BAND_NUM])
            band_data_list = context[Context.PRED_LIST]
            band_description_list = list(context[Context.BAND_DESCRIPTION_LIST])

            # Create new GeoTIFF raster - Can't use COG driver unless we use CreateCopy, which causes issues when we have less bands in SR than TOA
            ds_toa = gdal.Open(context[Context.FN_SRC])
            toa_ndv = ds_toa.GetRasterBand(1).GetNoDataValue()
            toa_datatype = ds_toa.GetRasterBand(1).DataType

            # Create new GeoTIFF raster - Can't use COG driver unless we use CreateCopy, which causes issues when we have less bands in SR than TOA
            driver_GTiff = gdal.GetDriverByName('GTiff')
            ds_toa_copy_GTiff = driver_GTiff.Create(intermediate_output_name, xsize=ds_toa.RasterXSize, ysize=ds_toa.RasterYSize, 
                                                    bands=numBandPairs, eType=toa_datatype, options=['COMPRESS=LZW','BIGTIFF=YES'])
            # ds_toa_copy_GTiff = driver_GTiff.Create(intermediate_output_name, xsize=ds_toa.RasterXSize, ysize=ds_toa.RasterYSize, 
            #                                         bands=numBandPairs, eType=toa_datatype, options=['COMPRESS=LZW'])
            
            # Set metadata to match TOA. 
            ds_toa_copy_GTiff.SetGeoTransform(ds_toa.GetGeoTransform())
            ds_toa_copy_GTiff.SetProjection(ds_toa.GetProjection())
    
            # Populate each output band with simulated prediction arrays         
            band_index_list = []
            band_data_list30m = []
            for id in range(0, numBandPairs):
                band_index_list.append(int(id+1))
                band = ds_toa_copy_GTiff.GetRasterBand(id+1)
                band.SetDescription(str(band_description_list[id]))
                band.SetNoDataValue(toa_ndv)
                bandPrediction2m = np.ma.masked_values(band_data_list[id], context[Context.TARGET_NODATA_VALUE])
                self._plot_lib.trace(f"Writing simulated prediction band...   {str(band_description_list[id])}")
                band.WriteArray(bandPrediction2m)

                # Downscale new band to 30m for statistics-generation later
                # bandPrediction30m = np.resize(bandPrediction2m, (757, 505))
                # band_data_list30m.append(bandPrediction30m)

            if (not (eval(context[Context.NONCOG_FLAG]))):
                translateoptions = gdal.TranslateOptions( format="COG", bandList=band_index_list,
                                                    creationOptions=['BIGTIFF=YES'])
                dsCog = gdal.Translate(context[Context.FN_DEST], ds_toa_copy_GTiff, options=translateoptions)
                dsCog = None

                self._plot_lib.trace(f"\nCreated COG from stack of regressed bands...\n   {context[Context.FN_COG]}")
            else:
                cog_output_name = os.path.join(context[Context.DIR_OUTPUT] + '/' +
                                               context[Context.FN_PREFIX] + str(Context.FN_SRLITE_SUFFIX))
                os.rename(intermediate_output_name, cog_output_name)
                self._plot_lib.trace(f"\nCreated standard TIF from stack of regressed bands...\n   {cog_output_name}")

        except BaseException as err:
            issue = "TIF file creation failed: " + str(err)
            self._plot_lib.trace(f"\nImage creation failed for: " + intermediate_output_name + "Issue = " + str(issue))
            raise Exception(issue)
        finally:
            # Remove pre-COG image
            fileExists = (os.path.exists(intermediate_output_name))
            if fileExists:
                self.removeFile(intermediate_output_name, context[Context.CLEAN_FLAG])
            ds_toa_copy_GTiff = None
            ds_toa = None
        
        # cheap attempt to avoid reprojection of TOA NONCOG from 2m to 30m for statistics calc
        # context['band_data_list30m'] = band_data_list30m
        return context[Context.FN_DEST]
    # -------------------------------------------------------------------------
    # removeFile()
    #
    # Optionally remove file from file system
    # -------------------------------------------------------------------------
    def removeFile(self, fileName, cleanFlag):

        if eval(cleanFlag):
            if os.path.exists(fileName):
                os.remove(fileName)

    # -------------------------------------------------------------------------
    # createCOG()
    #
    # Manage lifecycle of a Cloud-Optimized GeoTIF (COG)
    # -------------------------------------------------------------------------
    def createCOG(self, context):
        self._validateParms(context, [Context.FN_SRC, Context.CLEAN_FLAG,
                                      Context.FN_DEST])

        # Clean pre-COG image
        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        self.cog(context)
        # TBD - This is where noncog.tif gets cleaned, so droppings occur if processing has an error - fix this
        self.removeFile(context[Context.FN_SRC], 'True')

        return context[Context.FN_DEST]

    # -------------------------------------------------------------------------
    # _getExtents()
    #
    # Get extents from raster file
    # -------------------------------------------------------------------------
    def _getExtents(self, in_raster):
        # Get extents from raster
        data = gdal.Open(in_raster, gdal.GA_ReadOnly)
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        extent = [minx, miny, maxx, maxy]
        data = None
        self._plot_lib.trace("[ EXTENT ] = {}".format(extent))
        return extent

    # -------------------------------------------------------------------------
    # cog()
    #
    # Create a Cloud-Optimized GeoTIF (COG) from a TIF file
    # -------------------------------------------------------------------------
    def cog(self, context):
        self._validateParms(context, [Context.CLEAN_FLAG, Context.FN_DEST,
                                      Context.FN_SRC,
                                      Context.TARGET_XRES, Context.TARGET_YRES])

        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        translateoptions = gdal.TranslateOptions( format="COG",
                                       creationOptions=['BIGTIFF=YES'])
        ds = gdal.Translate(context[Context.FN_DEST], context[Context.FN_SRC], options=translateoptions)
        ds = None

    # -------------------------------------------------------------------------
    # _applyThreshold()
    #
    # Eliminate pixel values in band that lie outside of min/max threshold
    # -------------------------------------------------------------------------
    def _applyThreshold(self, min, max, bandMaArray):
        ########################################
        # Mask threshold values (e.g., (median - threshold) < range < (median + threshold)
        #  prior to generating common mask to reduce outliers ++++++[as per MC - 02/07/2022]
        ########################################
        self._plot_lib.trace(
            '======== Applying threshold algorithm to first EVHR Band (Assume Blue) ========================')
        bandMaThresholdMaxArray = np.ma.masked_where(bandMaArray > max, bandMaArray)
        bandMaThresholdRangeArray = np.ma.masked_where(bandMaThresholdMaxArray < min, bandMaThresholdMaxArray)
        self._plot_lib.trace(' threshold range median =' + str(np.ma.median(bandMaThresholdRangeArray)))
        return bandMaThresholdRangeArray

    # -------------------------------------------------------------------------
    # get_ndv()
    #
    # Get NoDataValue from TIF file
    # -------------------------------------------------------------------------
    def get_ndv(self, r_fn):
        with rasterio.open(r_fn) as src:
            return src.profile['nodata']

    # -------------------------------------------------------------------------
    # purge(dir, pattern)
    #
    # Delete files of a certain pattern in a specific directory
    # -------------------------------------------------------------------------
    def purge(self, dir, pattern):
        if (os.path.isdir(dir)):
            for f in os.listdir(dir):
                if re.search(pattern, f):
                    os.remove(os.path.join(dir, f))

    # -------------------------------------------------------------------------
    # refresh()
    #
    # Release memory
    # -------------------------------------------------------------------------
    def refresh(self, context):

        # Restore handles to file pool and reset internal flags
        if str(Context.DS_TOA_DOWNSCALE) in context:
            context[Context.DS_TOA_DOWNSCALE] = None
        if str(Context.DS_TARGET_DOWNSCALE) in context:
            context[Context.DS_TARGET_DOWNSCALE] = None
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            if str(Context.DS_CLOUDMASK_DOWNSCALE) in context:
                context[Context.DS_CLOUDMASK_DOWNSCALE] = None
            if str(Context.MA_CLOUDMASK_DOWNSCALE) in context:
                context[Context.MA_CLOUDMASK_DOWNSCALE] = None

        if str(Context.DS_LIST) in context:
            if str(Context.LIST_INDEX_TOA) in context:
                if (int(context[Context.LIST_INDEX_TOA]) > -1):
                    context[Context.DS_LIST][context[Context.LIST_INDEX_TOA]] = None
            if str(Context.LIST_INDEX_TARGET) in context:
                if (int(context[Context.LIST_INDEX_TARGET]) > -1):
                    context[Context.DS_LIST][context[Context.LIST_INDEX_TARGET]] = None
            if str(Context.LIST_INDEX_CLOUDMASK) in context:
                if (int(context[Context.LIST_INDEX_CLOUDMASK]) > -1):
                    context[Context.DS_LIST][context[Context.LIST_INDEX_CLOUDMASK]] = None
            context[Context.DS_LIST] = None

        if str(Context.MA_LIST) in context:
            if str(Context.LIST_INDEX_TOA) in context:
                if (int(context[Context.LIST_INDEX_TOA]) > -1):
                    context[Context.MA_LIST][context[Context.LIST_INDEX_TOA]] = None
            if str(Context.LIST_INDEX_TARGET) in context:
                if (int(context[Context.LIST_INDEX_TARGET]) > -1):
                    context[Context.MA_LIST][context[Context.LIST_INDEX_TARGET]] = None
            if str(Context.LIST_INDEX_CLOUDMASK) in context:
                if (int(context[Context.LIST_INDEX_CLOUDMASK]) > -1):
                    context[Context.MA_LIST][context[Context.LIST_INDEX_CLOUDMASK]] = None
            context[Context.MA_LIST] = None
