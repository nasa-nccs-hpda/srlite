#!/usr/bin/env python
# coding: utf-8
import os
import os.path
import sys
import ast
from datetime import datetime
import osgeo
from osgeo import gdal, osr
from pygeotools.lib import iolib, warplib, malib
import rasterio
import numpy as np
from srlite.model.Context import Context
from sklearn.linear_model import HuberRegressor, LinearRegression

#Not in current ilab kernel
from pylr2 import regress2
import pandas as pd

# -----------------------------------------------------------------------------
# class Context
#
# This class is a serializable context for orchestration.
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
            print('ERROR - check gdal version: ',err)
            sys.exit(1)
        return

    def _validateParms(self, context, requiredList):
        """
        Verify that required parameters exist in Context()
        """
        for parm in requiredList:
            parmExists = False
            for item in context:
                if (item == parm):
                    parmExists = True
                    break;
            if (parmExists == False):
                print("Error: Missing required parameter: " + str(parm))
                exit(1)

    def getBandIndices(self, context):
        self._validateParms(context, [Context.LIST_BAND_PAIRS, Context.FN_LIST])
        """
        Validate band name pairs and return corresponding gdal indices
        """
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        self._plot_lib.trace('bandNamePairList=' + str(bandNamePairList))

        fn_list = context[Context.FN_LIST]
        ccdcDs = gdal.Open(fn_list[context[Context.LIST_INDEX_TARGET]], gdal.GA_ReadOnly)
        ccdcBands = ccdcDs.RasterCount
        evhrDs = gdal.Open(fn_list[context[Context.LIST_INDEX_TOA]], gdal.GA_ReadOnly)
        evhrBands = evhrDs.RasterCount

        numBandPairs = len(bandNamePairList)
        bandIndices = [numBandPairs]
        toaBandNames = []

        for bandPairIndex in range(0, numBandPairs):

            ccdcBandIndex = evhrBandIndex = -1
            currentBandPair = bandNamePairList[bandPairIndex]

            for ccdcIndex in range(1, ccdcBands + 1):
                # read in bands from image
                band = ccdcDs.GetRasterBand(ccdcIndex)
                bandDescription = band.GetDescription()
                bandName = currentBandPair[context[Context.LIST_INDEX_TARGET]]
                if len(bandDescription) == 0:
                    ccdcBandIndex = bandPairIndex + 1
                    self._plot_lib.trace(f"Band has no description {bandName} - assume index of current band  {ccdcBandIndex}")
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
                if len(bandDescription) == 0:
                    evhrBandIndex = bandPairIndex + 1
                    self._plot_lib.trace(f"Band has no description {bandName} - assume index of current band  {evhrBandIndex}")
                    break
                else:
                    if (bandDescription == bandName):
                        evhrBandIndex = evhrIndex
                        break

            if ((ccdcBandIndex == -1) or (evhrBandIndex == -1)):
                ccdcDs = evhrDs = None
                self._plot_lib.trace(f"Invalid band pairs - verify correct name and case {currentBandPair}")
                exit(f"Invalid band pairs - verify correct name and case {currentBandPair}")

            bandIndices.append([ccdcIndex, evhrIndex])
            toaBandNames.append(currentBandPair[1])

        context[Context.LIST_TOA_BANDS] = toaBandNames

        ccdcDs = evhrDs = None
        self._plot_lib.trace('validated bandIndices=' + str(bandIndices))
        return bandIndices

    def getAttributeSnapshot(self, context):
        self._validateParms(context, [Context.FN_TOA, Context.FN_TARGET, Context.FN_CLOUDMASK])

        # Get snapshot of attributes of EVHR, CCDC, and Cloudmask tifs and create plot")
        self.getAttributes(str(context[Context.FN_TOA]), "EVHR Combo Plot")
        self.getAttributes(str(context[Context.FN_TARGET]), "CCDC Combo Plot")
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            self.getAttributes(str(context[Context.FN_CLOUDMASK]), "Cloudmask Combo Plot")

    def getAttributes(self, r_fn, title=None):
        geotransform = None
        r_ds = iolib.fn_getds(r_fn)
        if (self._debug_level >= 1):
            self._plot_lib.trace("\n File Name is {}".format(r_fn))
            self._plot_lib.trace(r_ds.GetProjection())
            self._plot_lib.trace("Driver: {}/{}".format(r_ds.GetDriver().ShortName,
                                         r_ds.GetDriver().LongName))
            self._plot_lib.trace("Size is {} x {} x {}".format(r_ds.RasterXSize,
                                                r_ds.RasterYSize,
                                                r_ds.RasterCount))
            self._plot_lib.trace("Projection is {}".format(r_ds.GetProjection()))
            geotransform = r_ds.GetGeoTransform()
            if geotransform:
                self._plot_lib.trace("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
                self._plot_lib.trace("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

        if (self._debug_level >= 2):
            self._plot_lib.plot_combo(r_fn, figsize=(14, 7), title=title)

        r_ds = None
        return geotransform

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

    def getIntersectionDs(self, context):
        self._validateParms(context, [Context.DS_INTERSECTION_LIST])

        # ########################################
        # # Align the CCDC and EVHR images, then take the intersection of the grid points
        # ########################################
        warp_ds_list = warplib.memwarp_multi(
            context[Context.DS_INTERSECTION_LIST], res='first', extent='intersection', t_srs='first', r=context[Context.TARGET_SAMPLING_METHOD])
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        self._plot_lib.trace('\n TOA shape=' + str(warp_ma_list[0].shape) + ' TARGET shape=' + str(warp_ma_list[1].shape))
        return warp_ds_list, warp_ma_list

    def applyCommonMask(self, context):
        self._validateParms(context, [Context.COMMON_MASK,  Context.MA_WARP_LIST])

        context[Context.MA_WARP_MASKED_LIST] =  \
            [np.ma.array(ma, mask=context[Context.COMMON_MASK]) for ma in  context[Context.MA_WARP_LIST]]

        #plot_maps(warp_ma_masked_list, fn_list, title_text='30 m warped version with a common mask applied\n')
        #plot_hists(warp_ma_masked_list, fn_list, title_text='30 m warped version with a common mask applied\n')
        return  context[Context.MA_WARP_MASKED_LIST]

    def applyCommonMask(self, context):
        self._validateParms(context, [Context.COMMON_MASK,  Context.MA_WARP_LIST])

        context[Context.MA_WARP_MASKED_LIST] =  \
            [np.ma.array(ma, mask=context[Context.COMMON_MASK]) for ma in  context[Context.MA_WARP_LIST]]

        #plot_maps(warp_ma_masked_list, fn_list, title_text='30 m warped version with a common mask applied\n')
        #plot_hists(warp_ma_masked_list, fn_list, title_text='30 m warped version with a common mask applied\n')
        return  context[Context.MA_WARP_MASKED_LIST]

    def getCommonMask(self, context):
        self._validateParms(context, [Context.MA_WARP_LIST])

        # ########################################
        # # Align the CCDC and EVHR images, then take the intersection of the grid points
        # ########################################
        # Mask negative values in input
        context[Context.MA_WARP_VALID_LIST] = \
            [np.ma.masked_where(ma < 0, ma) for ma in context[Context.MA_WARP_LIST]]

        common_mask = malib.common_mask(context[Context.MA_WARP_VALID_LIST])
        return common_mask

    def getMaskedArrays(self, context):
        self._validateParms(context, [Context.FN_LIST])

        # ########################################
        # # Align the CCDC and EVHR images, then take the intersection of the grid points
        # ########################################
        context[Context.MA_LIST]  = [iolib.fn_getma(fn) for fn in context[Context.FN_LIST]]
        return context[Context.MA_LIST]

    def maskNegativeValues(self, context):
        self._validateParms(context, [Context.MA_LIST])

        warp_valid_ma_list= [np.ma.masked_where(ma < 0, ma) for ma in context[Context.MA_LIST]]
        return warp_valid_ma_list

    def getIntersection(self, context):
        self._validateParms(context, [Context.FN_INTERSECTION_LIST])

        # ########################################
        # # Take the intersection of the scenes and return masked arrays of common pixels
        # ########################################
        # warp_ds_list = warplib.memwarp_multi_fn(
        #     context[Context.FN_INTERSECTION_LIST], res=context[Context.TARGET_XRES] , extent='intersection', t_srs='first', r=context[Context.TARGET_SAMPLING_METHOD])
        warp_ds_list = warplib.memwarp_multi_fn(
            context[Context.FN_INTERSECTION_LIST], res='first', extent='intersection', t_srs='first', r=context[Context.TARGET_SAMPLING_METHOD])
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        return warp_ds_list, warp_ma_list

    def getCcdcReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST,Context.FN_TARGET,Context.FN_TOA])

        # ########################################
        # # Align context[Context.FN_LIST[>0]] to context[Context.FN_LIST[0]] return masked arrays of reprojected pixels
        # ########################################
        warp_ds_list = warplib.memwarp_multi_fn(context[Context.FN_LIST],
                                                res=str(context[Context.FN_TARGET]),
                                                extent=str(context[Context.FN_TARGET]),
                                                t_srs=str(context[Context.FN_TARGET]),
                                                r='average',
                                                dst_ndv=self.get_ndv(str(context[Context.FN_TOA]),
                                                                     ))
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]

        # self._plot_lib.plot_maps2(warp_ma_list, context[Context.FN_LIST], title_text='30 m warped version\n')
        # self._plot_lib.plot_hists2(warp_ma_list, context[Context.FN_LIST], title_text='30 m warped version\n')

        ndv_list = [self.get_ndv(fn) for fn in context[Context.FN_LIST]]
        print ([ma.get_fill_value() for ma in warp_ma_list])
        print(ndv_list)

        return warp_ds_list, warp_ma_list

    def getReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST, Context.TARGET_FN, Context.TARGET_SAMPLING_METHOD])

        # ########################################
        # # Align context[Context.FN_LIST[>0]] to context[Context.FN_LIST[0]] return masked arrays of reprojected pixels
        # ########################################
        warp_ds_list = warplib.memwarp_multi_fn(context[Context.FN_LIST],
                                                res=context[Context.TARGET_XRES],
                                                extent=str(context[Context.TARGET_FN]),
                                                t_srs=str(context[Context.TARGET_FN]),
                                                r=context[Context.TARGET_SAMPLING_METHOD] ,
                                                dst_ndv=self.get_ndv(str(context[Context.TARGET_FN]),
                                                                     ))
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]

        # self._plot_lib.plot_maps2(warp_ma_list, context[Context.FN_LIST], title_text='30 m warped version\n')
        # self._plot_lib.plot_hists2(warp_ma_list, context[Context.FN_LIST], title_text='30 m warped version\n')

        ndv_list = [self.get_ndv(fn) for fn in context[Context.FN_LIST]]
        print([ma.get_fill_value() for ma in warp_ma_list])
        print(ndv_list)

        return warp_ds_list, warp_ma_list

    def __getReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST])

        # ########################################
        # # Align context[Context.FN_LIST[>0]] to context[Context.FN_LIST[0]] return masked arrays of reprojected pixels
        # ########################################
        warp_ds_list = warplib.memwarp_multi_fn(
            context[Context.FN_LIST], res=context[Context.TARGET_XRES] , extent='first', t_srs='first')
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        return warp_ds_list, warp_ma_list

    def applyEVHRCloudmask(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST])

        ccdc_warp_ma = context[Context.MA_WARP_LIST][0]
        evhr_warp_ma = context[Context.MA_WARP_LIST][1]
        cloudmask_warp_ma = context[Context.MA_WARP_LIST][2]
        context[Context.MA_WARP_CLOUD_LIST] = [ccdc_warp_ma, evhr_warp_ma,
                                        np.ma.masked_where(cloudmask_warp_ma == 1.0, cloudmask_warp_ma)]
        return context[Context.MA_WARP_CLOUD_LIST]

    def prepareEVHRCloudmask(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_INDEX_CLOUDMASK])

        # Mask out clouds
        cloudmask_warp_ma = context[Context.MA_WARP_CLOUD_LIST][0]
        cloudmaskWarpExternalBandMaArrayMasked = \
            np.ma.masked_where(cloudmask_warp_ma == 1.0, cloudmask_warp_ma)
        # cloudmaskWarpExternalBandMaArrayMasked = \
        #     np.ma.masked_where(cloudmask_warp_ma > 0.0, cloudmask_warp_ma)

        return cloudmaskWarpExternalBandMaArrayMasked

    def _prepareEVHRCloudmask(self, context):
        self._validateParms(context,
                            [Context.MA_CLOUDMASK_DOWNSCALE])

        cloudmaskWarpExternalBandMaArray = context[Context.MA_CLOUDMASK_DOWNSCALE]
        cloudmaskWarpExternalBandMaArraycloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(cloudmaskWarpExternalBandMaArray == 1.0,
                                                                    cloudmaskWarpExternalBandMaArray)
        return cloudmaskWarpExternalBandMaArraycloudmaskWarpExternalBandMaArrayMasked

    def _prepareEVHRCloudmask(self, context):
        self._validateParms(context,
                            [Context.DS_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])


        #  Get Masked array from warped Cloudmask - assumes only 1 band in mask to be applied to all
        cloudmaskWarpExternalBandMaArray = iolib.fn_getma(context[Context.FN_CLOUDMASK_DOWNSCALE], 1)
        self._plot_lib.trace(f'\nBefore Mask -> cloudmaskWarpExternalBandMaArray')
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray hist: {np.histogram(cloudmaskWarpExternalBandMaArray)}')
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray shape: {cloudmaskWarpExternalBandMaArray.shape}')
        count_non_masked = np.ma.count(cloudmaskWarpExternalBandMaArray)
        count_masked = np.ma.count_masked(cloudmaskWarpExternalBandMaArray)
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray ma.count (non-masked)=' + str(count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray ma.count_masked (masked)=' + str(count_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArray total count (masked + non-masked)=' + str(
                count_masked + count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray max=' + str(cloudmaskWarpExternalBandMaArray.max()))
        self._plot_lib.plot_combo_array(cloudmaskWarpExternalBandMaArray, figsize=(14, 7),
                                 title='cloudmaskWarpExternalBandMaArray')

        # Create a mask where the pixel values equal to 'one' are suppressed because these correspond to clouds
        self._plot_lib.trace(
            f'\nAfter Mask == 1.0 (sum should be 0 since all ones are masked -> cloudmaskWarpExternalBandMaArray')
        cloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(cloudmaskWarpExternalBandMaArray == 1.0,
                                                                    cloudmaskWarpExternalBandMaArray)
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked hist: {np.histogram(cloudmaskWarpExternalBandMaArrayMasked)}')
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked shape: {cloudmaskWarpExternalBandMaArrayMasked.shape}')
        count_non_masked = np.ma.count(cloudmaskWarpExternalBandMaArrayMasked)
        count_masked = np.ma.count_masked(cloudmaskWarpExternalBandMaArrayMasked)
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count (masked)=' + str(count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count_masked (non-masked)=' + str(count_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked total count (masked + non-masked)=' + str(
            count_masked + count_non_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked max=' + str(cloudmaskWarpExternalBandMaArrayMasked.max()))
        self._plot_lib.plot_combo_array(cloudmaskWarpExternalBandMaArrayMasked, figsize=(14, 7),
                                 title='cloudmaskWarpExternalBandMaArrayMasked')

        self.removeFile(context[Context.FN_CLOUDMASK_DOWNSCALE], str('True'))

        return cloudmaskWarpExternalBandMaArrayMasked

    def prepareQualityFlagMask(self, context):
        self._validateParms(context,
                            [Context.DS_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        #  Get Masked array from warped Cloudmask - get Band 8 (https://glad.umd.edu/Potapov/ARD/ARD_manual_v1.1.pdf)
        cloudmaskWarpExternalBandMaArray = iolib.fn_getma(context[Context.FN_TARGET_DOWNSCALE], 8)
        self._plot_lib.trace(f'\nBefore Mask -> cloudmaskWarpExternalBandMaArray')
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray hist: {np.histogram(cloudmaskWarpExternalBandMaArray)}')
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray shape: {cloudmaskWarpExternalBandMaArray.shape}')
        count_non_masked = np.ma.count(cloudmaskWarpExternalBandMaArray)
        count_masked = np.ma.count_masked(cloudmaskWarpExternalBandMaArray)
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray ma.count (masked)=' + str(count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray ma.count_masked (non-masked)=' + str(count_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArray total count (masked + non-masked)=' + str(
                count_masked + count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray max=' + str(cloudmaskWarpExternalBandMaArray.max()))
        self._plot_lib.plot_combo_array(cloudmaskWarpExternalBandMaArray, figsize=(14, 7),
                                 title='cloudmaskWarpExternalBandMaArray')

        # Create a mask where the pixel values equal to '0, 3, 4' are suppressed because these correspond to NoData, Clouds, and Cloud Shadows
        self._plot_lib.trace(
            f'\nSuppress values=[0, 3, 4] according to Band #8 because they correspond to NoData, Clouds, and Cloud Shadows')
#        cloudmaskWarpExternalBandMaArrayMasked = (cloudmaskWarpExternalBandMaArray == 0) & (cloudmaskWarpExternalBandMaArray == 3) & (cloudmaskWarpExternalBandMaArray == 4)

        ndv = int(Context.DEFAULT_NODATA_VALUE)
        cloudmaskWarpExternalBandMaArrayData = np.ma.getdata(cloudmaskWarpExternalBandMaArray)
        cloudmaskWarpExternalBandMaArrayData = np.select([cloudmaskWarpExternalBandMaArrayData == 0, cloudmaskWarpExternalBandMaArrayData == 3,
                           cloudmaskWarpExternalBandMaArrayData == 4], [ndv, ndv, ndv], cloudmaskWarpExternalBandMaArrayData)
        cloudmaskWarpExternalBandMaArrayData = np.select([cloudmaskWarpExternalBandMaArrayData != ndv], [0.0], cloudmaskWarpExternalBandMaArrayData)
        cloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(cloudmaskWarpExternalBandMaArrayData == ndv, cloudmaskWarpExternalBandMaArrayData)

        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked hist: {np.histogram(cloudmaskWarpExternalBandMaArrayMasked)}')
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked shape: {cloudmaskWarpExternalBandMaArrayMasked.shape}')
        count_non_masked = np.ma.count(cloudmaskWarpExternalBandMaArrayMasked)
        count_masked = np.ma.count_masked(cloudmaskWarpExternalBandMaArrayMasked)
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count (masked)=' + str(count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count_masked (non-masked)=' + str(count_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMasked total count (masked + non-masked)=' + str(
            count_masked + count_non_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked max=' + str(cloudmaskWarpExternalBandMaArrayMasked.max()))
        self._plot_lib.plot_combo_array(cloudmaskWarpExternalBandMaArrayMasked, figsize=(14, 7),
                                 title='cloudmaskWarpExternalBandMaArrayMasked')
        return cloudmaskWarpExternalBandMaArrayMasked

    def ma2df(self, ma, product, band):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False]
        df = pd.DataFrame(unmasked)
        df.columns = [product + band]
        df[product + band] = df[product + band] * 0.0001
        # df.columns = ['Reflectance']
        # df['Product'] = product
        # df['Band'] = band
        return df

    def performRegression(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]
        self._plot_lib.trace('bandPairIndicesList: ' + str(bandPairIndicesList))
        numBandPairs = len(bandPairIndicesList)

        warp_ma_masked_band_series = [numBandPairs]
        sr_prediction_list = []
        warp_ds_list = context[Context.DS_WARP_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        minWarning = 0
        firstBand = True

        cloudmaskEVHRWarpExternalBandMaArrayMasked = cloudmaskQFWarpExternalBandMaArrayMasked = None
        # Get optional Cloudmask
        cloudMask = False
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            cloudMask = True
            cloudmaskEVHRWarpExternalBandMaArrayMasked = self.prepareEVHRCloudmask(context)

        # Get optional Quality flag mask
        qfMask = False
        if (eval(context[Context.QUALITY_MASK_FLAG] )):
            qfMask = True
            cloudmaskQFWarpExternalBandMaArrayMasked = self.prepareQualityFlagMask(context)

        # Get optional Threshold mask
        thMask = False
        if (eval(context[Context.THRESHOLD_MASK_FLAG])):
            thMask = True
              # Apply user-specified threshold range for Blue Band pixel mask and apply to each band
            evhrBandMaArrayThresholdMin = context[Context.THRESHOLD_MIN]
            evhrBandMaArrayThresholdMax = context[Context.THRESHOLD_MAX]
            self._plot_lib.trace(' evhrBandMaArrayThresholdMin = ' + str(evhrBandMaArrayThresholdMin))
            self._plot_lib.trace(' evhrBandMaArrayThresholdMax = ' + str(evhrBandMaArrayThresholdMax))
            minWarning = evhrBandMaArrayThresholdMin

        ########################################
        # ### FOR EACH BAND PAIR,
        # now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
        ########################################
        for bandPairIndex in range(0, numBandPairs - 1):

            self._plot_lib.trace('=>')
            self._plot_lib.trace('====================================================================================')
            self._plot_lib.trace('============== Start Processing Band #' + str(bandPairIndex + 1) + ' ===============')
            self._plot_lib.trace('====================================================================================')

            # Retrieve band pair
            bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

            # Get 30m CCDC Masked Arrays
#           cloudmask_warp_ma = context[Context.MA_WARP_LIST][context[Context.LIST_INDEX_CLOUDMASK]]
#            ccdcBandMaArray = [context[Context.MA_WARP_LIST][context[Context.LIST_INDEX_TARGET]]][bandPairIndices[context[Context.LIST_INDEX_TARGET]
 #           evhrBandMaArray = [context[Context.MA_WARP_LIST][context[Context.LIST_INDEX_TOA]]][bandPairIndices[context[Context.LIST_INDEX_TARGET]
#            evhrBandMaArray = context[Context.MA_WARP_LIST][context[Context.LIST_INDEX_TOA]]
#              ccdcBandMaArray = iolib.ds_getma(warp_ds_list[context[Context.LIST_INDEX_TARGET]],
#                                               bandPairIndices[context[Context.LIST_INDEX_TARGET]])
#              evhrBandMaArray = iolib.ds_getma(warp_ds_list[context[Context.LIST_INDEX_TOA]],
#                                               bandPairIndices[context[Context.LIST_INDEX_TOA]])
            ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
            evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

            #  Create single mask for all bands based on Blue-band threshold values
            #  Assumes Blue-band is first indice pair, so collect mask on 1st iteration only.
            if (thMask == True):
                if (firstBand == True):
                    evhrBandMaThresholdArray = self._applyThreshold(evhrBandMaArrayThresholdMin, evhrBandMaArrayThresholdMax,
                                                               evhrBandMaArray)
                    firstBand = False

            #  Create a common mask that intersects the CCDC/QF, EVHR, and Cloudmasks - this will then be used to correct the input EVHR & CCDC/QF
            warp_ma_band_list_all = [ccdcBandMaArray, evhrBandMaArray]
            if (cloudMask == True):
                warp_ma_band_list_all.append(cloudmaskEVHRWarpExternalBandMaArrayMasked)
            if (qfMask == True):
                warp_ma_band_list_all.append(cloudmaskQFWarpExternalBandMaArrayMasked)
            if (thMask == True):
                evhrBandMaArray = evhrBandMaThresholdArray
                warp_ma_band_list_all.append(evhrBandMaThresholdArray)

            # TODO fix diagnostics
            # [print(ma.shape) for ma in warp_ma_band_list_all]
            # print([warp_ma.get_fill_value() for warp_ma in warp_ma_band_list_all])
            # print('Input array mins/maxs')
            # [print(f'input ma min: {ma.min()}') for ma in context[Context.MA_LIST]]
            # [print(f'input ma max: {ma.max()}') for ma in context[Context.MA_LIST]]
            # print('Warped array mins/maxs')
            # [print(f'warped ma min: {ma.min()}') for ma in warp_ma_band_list_all]
            # [print(f'warped ma max: {ma.max()}') for ma in warp_ma_band_list_all]
            #
            # Mask negative values in input
            # TODO make negative value masking optional
            warp_valid_ma_band_list_all = [np.ma.masked_where(ma < 0, ma) for ma in warp_ma_band_list_all]

            # Create common mask
            common_mask_band_all = malib.common_mask(warp_valid_ma_band_list_all)

            # Apply the 3-way common mask to the CCDC and EVHR bands (effectively dropping unneeded Cloudmask)
            warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band_all),
                                        np.ma.array(evhrBandMaArray, mask=common_mask_band_all)]

            # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
#                    exit(1)
            self._plot_lib.plot_maps(warp_ma_masked_band_list, context[Context.FN_LIST], figsize=(10, 5),
                              title=str(bandNamePairList[bandPairIndex]) + ' Reflectance (%)')
            self._plot_lib.plot_histograms(warp_ma_masked_band_list, context[Context.FN_LIST], figsize=(10, 3),
                                    title=str(bandNamePairList[bandPairIndex]) + " BAND COMMON ARRAY")

            warp_ma_masked_band_series.append(warp_ma_masked_band_list)

            ########################################
            # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
            # CCDC SR is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
            ########################################

            ccdc_sr_index = context[Context.LIST_INDEX_TARGET]
            ccdc_sr_band = warp_ma_masked_band_list[ccdc_sr_index].ravel()
            evhr_sr_index = context[Context.LIST_INDEX_TOA]
            evhr_toa_band = warp_ma_masked_band_list[evhr_sr_index].ravel()

            ccdc_sr_data_only_band = ccdc_sr_band[ccdc_sr_band.mask == False]
            ccdc_sr_data_only_band_reshaped = ccdc_sr_data_only_band.reshape(-1, 1)
            evhr_toa_data_only_band = evhr_toa_band[evhr_toa_band.mask == False]
            evhr_toa_data_only_band_reshaped = evhr_toa_data_only_band.reshape(-1, 1)

            self._plot_lib.trace('Check the mins of the input and output...')
            input_min = warp_ma_masked_band_list[evhr_sr_index].reshape(-1, 1).min()
            self._plot_lib.trace("Check input TOA min: " + str(input_min))

            # Perform regression fit based on model type (TARGET against TOA)
            if (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_ROBUST):
                model_data_only_band = HuberRegressor().fit(
                    evhr_toa_data_only_band_reshaped, ccdc_sr_data_only_band_reshaped )
                self._plot_lib.trace(str(bandNamePairList[bandPairIndex]) + '= > intercept: ' + str(
                    model_data_only_band.intercept_) + ' slope: ' + str(model_data_only_band.coef_) + ' score: ' +
                                     str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1),
                                                                    ccdc_sr_data_only_band)))
                self._plot_lib.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                                        model_data_only_band.intercept_)
                self._plot_lib.trace("Check output prediction of SR for the input TOA min: " +
                                     str(model_data_only_band.predict(input_min.reshape(-1, 1))))
            elif (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_SIMPLE):
                model_data_only_band = LinearRegression().fit(
                    evhr_toa_data_only_band_reshaped, ccdc_sr_data_only_band_reshaped )
                self._plot_lib.trace(str(bandNamePairList[bandPairIndex]) + '= > intercept: ' + str(
                    model_data_only_band.intercept_) + ' slope: ' + str(model_data_only_band.coef_) + ' score: ' +
                                     str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1),
                                                                    ccdc_sr_data_only_band)))
                self._plot_lib.trace("Check output prediction of SR for the input TOA min: " +
                                     str(model_data_only_band.predict(input_min.reshape(-1, 1))))
                self._plot_lib.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                                        model_data_only_band.intercept_)
            elif (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_RMA):
#                rma_model = regress2(np.array(x_var), np.array(y_var), _method_type_2="reduced major axis")
                reflect_df = pd.concat([
                    self.ma2df(evhr_toa_data_only_band, 'EVHR_TOA', 'Band'),
                    self.ma2df(ccdc_sr_data_only_band, 'CCDC_SR', 'Band')],
                    axis=1)
                model_data_only_band = regress2(np.array(reflect_df['EVHR_TOABand']), np.array(reflect_df['CCDC_SRBand']),
                                         _method_type_2="reduced major axis")
#                model_data_only_band = regress2(np.array(evhr_toa_data_only_band),
 #                                               np.array(ccdc_sr_data_only_band),  _method_type_2="reduced major axis")
                rma_slope = model_data_only_band['slope']
                rma_intercept = model_data_only_band['intercept']
                self._plot_lib.trace(f'RMA Minimum Prediction Band slope intercept: {str(bandNamePairList[bandPairIndex])} {rma_slope} {rma_intercept}')
                minReshaped = (input_min.reshape(-1, 1))
                sr_prediction_min_band = (minReshaped * rma_slope) + (rma_intercept * 10000)
                self._plot_lib.trace("Check output prediction of SR for the input TOA min: " +
                                     str(sr_prediction_min_band))
                # self._plot_lib.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                #                         model_data_only_band.intercept_)
            else:
                print('Invalid regressor specified %s' % context[Context.REGRESSION_MODEL] )
                sys.exit(1)



            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file {os.path.basename(context[Context.FN_LIST][1])}')
            self._plot_lib.trace(f'Input masked array shape: {evhrBandMaArray.shape}')

            # score = model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
            # self._plot_lib.trace(f'R2 score : {score}')

            # Get 2m EVHR Masked Arrays

            ######## Double-checked this after Paul's srlite_warp_example Notebook
            evhrBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA], bandPairIndices[evhr_sr_index])
            evhrBandMaArrayRawReshaped = evhrBandMaArrayRaw.ravel().reshape(-1, 1)

 #           evhrBandMaArrayRaw = iolib.fn_getma(context[Context.FN_LIST][1], bandPairIndices[1])
            if (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_RMA):
                #evhr_srlite_rma_b = (warp_ma_masked_list[4] * rma_slope_b) + (rma_intercept_b * 10000)
                rma_slope = model_data_only_band['slope']
                rma_intercept = model_data_only_band['intercept']
                self._plot_lib.trace(f'RMA Band slope intercept: {str(bandNamePairList[bandPairIndex])} {rma_slope} {rma_intercept}')
                sr_prediction_band = (evhrBandMaArrayRaw * rma_slope) + (rma_intercept * 10000)
            else:
                sr_prediction_band = model_data_only_band.predict(evhrBandMaArrayRawReshaped)

            self._plot_lib.trace(f'Post-prediction shape : {sr_prediction_band.shape}')

            # Return to original shape and apply original mask
            orig_dims = evhrBandMaArrayRaw.shape
            evhr_sr_ma_band = np.ma.array(sr_prediction_band.reshape(orig_dims), mask=evhrBandMaArrayRaw.mask)

            # Check resulting ma
            self._plot_lib.trace(f'Final masked array shape: {evhr_sr_ma_band.shape}')

            ########### save prediction #############
            sr_prediction_list.append(evhr_sr_ma_band)

            ########################################
            ##### Compare the before and after histograms (EVHR TOA vs EVHR SR)
            ########################################
            evhr_pre_post_ma_list = [evhrBandMaArrayRaw, evhr_sr_ma_band]
            compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']
 #           self._plot_lib.plot_compare(evhr_pre_post_ma_list, compare_name_list)
 #           self._plot_lib.plot_maps2(evhr_pre_post_ma_list, compare_name_list, (10,50))

            # self._plot_lib.plot_histograms(evhr_pre_post_ma_list, context[Context.FN_LIST], figsize=(5, 3),
            #                         title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR")
            # self._plot_lib.plot_maps(evhr_pre_post_ma_list, compare_name_list, figsize=(10, 50))

            ########################################
            ##### Compare the original CCDC histogram with result (CCDC SR vs EVHR SR)
            ########################################
            #     ccdc_evhr_srlite_list = [ccdc_warp_ma, evhr_sr_ma_band]
            #     compare_name_list = ['CCDC SR', 'EVHR SR-Lite']

            #     self._plot_lib.plot_histograms(ccdc_evhr_srlite_list, fn_list, figsize=(5, 3),
            #                        title=str(bandNamePairList[bandPairIndex]) + " CCDC SR vs EVHR SR", override=override)
            #     self._plot_lib.plot_maps(ccdc_evhr_srlite_list, compare_name_list, figsize=(10, 50), override=override)

            ########################################
            ##### Compare the original EVHR TOA histogram with result (EVHR TOA vs EVHR SR)
            ########################################
            if False:
                evhr_srlite_delta_list = [evhr_pre_post_ma_list[1], evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]]
                compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']
                self._plot_lib.plot_histograms(evhr_srlite_delta_list, context[Context.FN_LIST], figsize=(5, 3),
                                        title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR DELTA ")
                self._plot_lib.plot_maps([evhr_pre_post_ma_list[1],
                                   evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]],
                                  [compare_name_list[1],
                                   str(bandNamePairList[bandPairIndex]) + ' Difference: TOA-SR-Lite'], (10, 50),
                                  cmap_list=['RdYlGn', 'RdBu'])
            print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

        return sr_prediction_list


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
        self._plot_lib.trace(f"\nAppy coefficients to High Res File...\n   {str(context[Context.FN_SRC])}")

        now = datetime.now()  # current date and time

        context[Context.FN_SUFFIX] = str(Context.FN_SRLITE_NONCOG_SUFFIX)
        context[Context.BAND_NUM] = len(list(context[Context.LIST_TOA_BANDS]))
        context[Context.BAND_DESCRIPTION_LIST] = list(context[Context.LIST_TOA_BANDS])
        context[Context.COG_FLAG] = True
        context[Context.TARGET_NODATA_VALUE] = int(Context.DEFAULT_NODATA_VALUE)

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
#            "dtype": context[Context.TARGET_DTYPE],
            "nodata": context[Context.TARGET_NODATA_VALUE],
            "descriptions": context[Context.BAND_DESCRIPTION_LIST]
        })

        band_data_list = context[Context.PRED_LIST]
#        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        band_description_list = list(context[Context.BAND_DESCRIPTION_LIST])

        ########################################
        # Read each layer and write it to stack
        ########################################
        with rasterio.open(output_name, 'w', **meta) as dst:
            for id in range(0, numBandPairs):
                bandPrediction = band_data_list[id]
                dst.set_band_description(id+1, band_description_list[id])
                bandPrediction1 = np.ma.masked_values(bandPrediction, context[Context.TARGET_NODATA_VALUE])
                dst.write_band(id+1, bandPrediction1)

        if (context[Context.COG_FLAG]):
            # Create Cloud-optimized Geotiff (COG)
            context[Context.FN_SRC] = str(output_name)
            context[Context.FN_DEST] = "{}/{}".format(
                context[Context.DIR_OUTPUT], str(context[Context.FN_PREFIX])
            ) + str(Context.FN_SRLITE_SUFFIX)
            cog_name = self.createCOG(context)

        self._plot_lib.trace(f"\nCreated COG from stack of regressed bands...\n   {cog_name}")
        return cog_name

    def removeFile(self, fileName, cleanFlag):

        if eval(cleanFlag):
            if os.path.exists(fileName):
                os.remove(fileName)

    def createCOG(self, context):
        self._validateParms(context, [Context.FN_SRC, Context.CLEAN_FLAG,
                            Context.FN_DEST])

        # Clean pre-COG image
        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        self.cog(context)
        self.removeFile(context[Context.FN_SRC], context[Context.CLEAN_FLAG])
        
        return context[Context.FN_DEST]

    def _getProjSrs(self, in_raster):
        # Get projection from raster
        ds = gdal.Open(in_raster)
        prj = ds.GetProjection()
        self._plot_lib.trace("[ PRJ ] = {}".format(prj))

        srs = osr.SpatialReference(wkt=prj)
        self._plot_lib.trace("[ SRS ] = {}".format(srs))
        if srs.IsProjected:
            self._plot_lib.trace("[ SRS.GetAttrValue('projcs') ] = {}".format(srs.GetAttrValue('projcs')))
        self._plot_lib.trace("[ SRS.GetAttrValue('geogcs') ] = {}".format(srs.GetAttrValue('geogcs')))
        return prj, srs

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

    def _getMetadata(self, band_num, input_file):
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

        self._plot_lib.trace("[ METADATA] = {}".format(src_ds.GetMetadata()))

        stats = srcband.GetStatistics(True, True)
        self._plot_lib.trace("[ STATS ] = Minimum={}, Maximum={}, Mean={}, StdDev={}".format( stats[0], stats[1], stats[2], stats[3]))

        self._plot_lib.trace("[ NO DATA VALUE ] = {}".format(srcband.GetNoDataValue()))
        self._plot_lib.trace("[ MIN ] = {}".format(srcband.GetMinimum()))
        self._plot_lib.trace("[ MAX ] = {}".format(srcband.GetMaximum()))
        self._plot_lib.trace("[ SCALE ] = {}".format(srcband.GetScale()))
        self._plot_lib.trace("[ UNIT TYPE ] = {}".format(srcband.GetUnitType()))
        ctable = srcband.GetColorTable()

        if ctable is None:
            self._plot_lib.trace('No ColorTable found')
            # sys.exit(1)
        else:
            self._plot_lib.trace("[ COLOR TABLE COUNT ] = ", ctable.GetCount())
            for i in range(0, ctable.GetCount()):
                entry = ctable.GetColorEntry(i)
                if not entry:
                    continue
                self._plot_lib.trace("[ COLOR ENTRY RGB ] = ", ctable.GetColorEntryAsRGB(i, entry))

        outputType = gdal.GetDataTypeName(srcband.DataType)

        self._plot_lib.trace(outputType)

        return srcband.DataType

    def warp(self, context):
        self._validateParms(context, [Context.CLEAN_FLAG, Context.FN_DEST,
                                      Context.TARGET_ATTR, Context.FN_DEST,
                                      Context.FN_SRC,
                                      Context.TARGET_SRS, Context.TARGET_OUTPUT_TYPE,
                                      Context.TARGET_XRES, Context.TARGET_YRES])

        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        extent = self._getExtents(context[Context.TARGET_ATTR])
        ds = gdal.Warp(context[Context.FN_DEST], context[Context.FN_SRC],
                       dstSRS=context[Context.TARGET_SRS] , outputType=context[Context.TARGET_OUTPUT_TYPE] ,
                       xRes=context[Context.TARGET_XRES] , yRes=context[Context.TARGET_YRES], outputBounds=extent)
        ds = None

    def downscale(self, context):
        self._validateParms(context, [Context.CLEAN_FLAG, Context.FN_DEST,
                                      Context.FN_SRC])

        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        if not (os.path.exists(context[Context.FN_DEST])):
            self.translate(context)
        self.getAttributes(str(context[Context.FN_DEST]), "Downscale Combo Plot")

    def translate(self, context):
        self._validateParms(context, [Context.CLEAN_FLAG, Context.FN_DEST,
                                      Context.FN_SRC,
                                      Context.TARGET_XRES, Context.TARGET_YRES])

        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        ds = gdal.Translate(context[Context.FN_DEST], context[Context.FN_SRC],
                            xRes=context[Context.TARGET_XRES], yRes=context[Context.TARGET_YRES])
        ds = None

    def cog(self, context):
        self._validateParms(context, [Context.CLEAN_FLAG, Context.FN_DEST,
                                      Context.FN_SRC,
                                      Context.TARGET_XRES, Context.TARGET_YRES])

        self.removeFile(context[Context.FN_DEST], context[Context.CLEAN_FLAG])
        ds = gdal.Translate(context[Context.FN_DEST], context[Context.FN_SRC], format="COG")
        ds = None

    def _applyThreshold(self, min, max, bandMaArray):
        ########################################
        # Mask threshold values (e.g., (median - threshold) < range < (median + threshold)
        #  prior to generating common mask to reduce outliers ++++++[as per MC - 02/07/2022]
        ########################################
        self._plot_lib.trace('======== Applying threshold algorithm to first EVHR Band (Assume Blue) ========================')
        bandMaThresholdMaxArray = np.ma.masked_where(bandMaArray > max, bandMaArray)
        bandMaThresholdRangeArray = np.ma.masked_where(bandMaThresholdMaxArray < min, bandMaThresholdMaxArray)
        self._plot_lib.trace(' threshold range median =' + str(np.ma.median(bandMaThresholdRangeArray)))
        return bandMaThresholdRangeArray

    def get_ndv(self, r_fn):
        with rasterio.open(r_fn) as src:
            return src.profile['nodata']

    def refresh(self, context):

        #Restore handles to file pool and reset internal flags
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

