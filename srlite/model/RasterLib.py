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

#Not in current ilab kernel (or introduced from diagnostics
from pylr2 import regress2
import pandas as pd
import sklearn

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

    def _representsInt(self, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    def getBandIndices(self, context):
        self._validateParms(context, [Context.LIST_BAND_PAIRS, Context.FN_LIST])
        """
        Validate band name pairs and return corresponding gdal indices
        """
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
                if (self._representsInt(bandName)):
                    evhrBandIndex = int(bandName)
                    break
                else:
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

            bandIndices.append([ccdcBandIndex, evhrBandIndex])
            toaBandNames.append(currentBandPair[1])
            targetBandNames.append(currentBandPair[0])

        context[Context.LIST_TOA_BANDS] = toaBandNames
        context[Context.LIST_TARGET_BANDS] = targetBandNames

        ccdcDs = evhrDs = None
        self._plot_lib.trace(f'Band Names: {str(bandNamePairList)} Indices: {str(bandIndices)}')
        return bandIndices

    def getAttributeSnapshot(self, context):
        self._validateParms(context, [Context.FN_TOA, Context.FN_TARGET, Context.FN_CLOUDMASK])

        # Get snapshot of attributes of EVHR, CCDC, and Cloudmask tifs and create plot")
        self.getAttributes(str(context[Context.FN_TOA]), "EVHR Combo Plot")
        self.getAttributes(str(context[Context.FN_TARGET]), "CCDC Combo Plot")
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            self.getAttributes(str(context[Context.FN_CLOUDMASK]), "Cloudmask Combo Plot")

    def replaceNdv(self, src_fn, new_ndv):
        ds = gdal.Open(src_fn)
        b = ds.GetRasterBand(1)
        #Extract old ndv
        old_ndv = iolib.get_ndv_b(b)

        print(src_fn)
        print("Replacing old ndv %s with new ndv %s" % (old_ndv, new_ndv))

        #Load masked array
        bma = iolib.ds_getma(ds)

        #Handle cases with input ndv of nan
        #if old_ndv == np.nan:
        bma = np.ma.fix_invalid(bma)

        #Set new fill value
        bma.set_fill_value(new_ndv)
        #Fill ma with new value and write out
        out_fn = os.path.splitext(src_fn)[0] + '_ndv.tif'
        iolib.writeGTiff(bma.filled(), out_fn, ds, ndv=new_ndv)
        return out_fn

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
                self._plot_lib.trace(f'Origin: ({geotransform[0]}, {geotransform[3]}), Resolution: ({geotransform[1]}, {geotransform[5]})  ')
                # self._plot_lib.trace("Origin = ({}, {})".format(geotransform[0], geotransform[3])) \
                # + "Resolution= ({}, {})".format(geotransform[1], geotransform[5])

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
        self._validateParms(context, [Context.FN_LIST, Context.FN_REPROJECTION_LIST, Context.TARGET_FN, Context.TARGET_SAMPLING_METHOD])

        # ########################################
        # # Align context[Context.FN_LIST[>0]] to context[Context.FN_LIST[0]] return masked arrays of reprojected pixels
        # ########################################
        ndv_list = [self.get_ndv(fn) for fn in context[Context.FN_REPROJECTION_LIST]]
        self._plot_lib.trace(f'Fill values before re-projection:  {ndv_list}')

    # Ensure that all NoData values match TARGET_FN (e.g., TOA)
        dst_ndv = self.get_ndv(str(context[Context.TARGET_FN]))
#        index = 0
        for fn in context[Context.FN_REPROJECTION_LIST]:
            current_ndv = self.get_ndv(fn)
            if (current_ndv != dst_ndv):
                out_fn = self.replaceNdv(fn, dst_ndv)
                index = context[Context.FN_LIST].index(fn)
                context[Context.FN_LIST][index] = out_fn
                index = context[Context.FN_REPROJECTION_LIST].index(fn)
                context[Context.FN_REPROJECTION_LIST][index] = out_fn
#               index = index+1

        # Reproject inputs to TOA attributes (res, extent, srs, nodata)
        warp_ds_list = warplib.memwarp_multi_fn(context[Context.FN_REPROJECTION_LIST],
                                                res=context[Context.TARGET_XRES],
                                                extent=str(context[Context.TARGET_FN]),
                                                t_srs=str(context[Context.TARGET_FN]),
                                                r=context[Context.TARGET_SAMPLING_METHOD],
                                                dst_ndv=dst_ndv)

        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        self._plot_lib.trace(f'Fill values after re-projection:  { [ma.get_fill_value() for ma in warp_ma_list]}')

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

    def prepareMasks(self, context):

        # Get optional Cloudmask
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            context['cloudmaskEVHRWarpExternalBandMaArrayMasked'] = self.prepareEVHRCloudmask(context)

        # Get optional Quality flag mask
        if (eval(context[Context.QUALITY_MASK_FLAG])):
            context['cloudmaskQFWarpExternalBandMaArrayMasked'] = self.prepareQualityFlagMask(context)

    def _getCommonMask(self, context, targetBandArray, toaBandArray):

        context['evhrBandMaThresholdArray'] = None
        #  Create a common mask that intersects the CCDC/QF, EVHR, and Cloudmasks - this will then be used to correct the input EVHR & CCDC/QF
        warp_ma_band_list_all = [targetBandArray, toaBandArray]
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            warp_ma_band_list_all.append(context['cloudmaskEVHRWarpExternalBandMaArrayMasked'])
        if (eval(context[Context.QUALITY_MASK_FLAG])):
            warp_ma_band_list_all.append(context['cloudmaskQFWarpExternalBandMaArrayMasked'])
        if (eval(context[Context.THRESHOLD_MASK_FLAG])):
            if (context['evhrBandMaThresholdArray']  == None):
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

    def predictSurfaceReflectance(self, context, band_name, toa_hr_band, target_sr_band, toa_sr_band):

        # Perform regression fit based on model type (TARGET against TOA)
        target_sr_band = target_sr_band.ravel()
        toa_sr_band = toa_sr_band.ravel()
        sr_prediction_band = None
        model_data_only_band = None

        target_sr_data_only_band = target_sr_band[target_sr_band.mask == False]
        target_sr_data_only_band_reshaped = target_sr_data_only_band.reshape(-1, 1)
        toa_sr_data_only_band = toa_sr_band[toa_sr_band.mask == False]
        toa_sr_data_only_band_reshaped = toa_sr_data_only_band.reshape(-1, 1)

        ####################
        ### Huber (robust) Regressor
        ####################
        if (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_ROBUST):
            # ravel the Y band (e.g., CCDC) - /home/gtamkin/.conda/envs/ilab_gt/lib/python3.7/site-packages/sklearn/utils/validation.py:993: DataConversion
            # Warning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
            model_data_only_band = HuberRegressor().fit(
                toa_sr_data_only_band_reshaped, target_sr_data_only_band_reshaped.ravel())

            #  band-specific metadata
            intercept = model_data_only_band.intercept_
            slope = model_data_only_band.coef_
            score = model_data_only_band.score(toa_sr_data_only_band.reshape(-1, 1), target_sr_data_only_band)
            sr_prediction_band = model_data_only_band.predict(toa_hr_band.reshape(-1, 1))

        ####################
        ### OLS (simple) Regressor
        ####################
        elif (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_MODEL_SIMPLE):
            model_data_only_band = LinearRegression().fit(
                toa_sr_data_only_band_reshaped, target_sr_data_only_band_reshaped)

            #  band-specific metadata
            intercept = model_data_only_band.intercept_
            slope = model_data_only_band.coef_
            score = model_data_only_band.score(toa_sr_data_only_band.reshape(-1, 1), target_sr_data_only_band)
            sr_prediction_band = model_data_only_band.predict(toa_hr_band.reshape(-1, 1))

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

            slope = model_data_only_band['slope']
            intercept = model_data_only_band['intercept']

            sr_prediction_band = (toa_hr_band * slope) + (intercept * 10000)

        else:
            print('Invalid regressor specified %s' % context[Context.REGRESSION_MODEL])
            sys.exit(1)

        metrics_srlite = None
        if (metrics_srlite != None):
            reflect_df = pd.concat([
                self.ma2df(target_sr_band,'CCDC_SR', 'Band-B'),
                self.ma2df(toa_sr_band, 'EVHR_TOA', 'Band-B')],
                axis=1)
            reflect_df_long = pd.wide_to_long(reflect_df.reset_index(),
                                              stubnames=['CCDC_SR', 'EVHR_TOA'],
                                              #  stubnames=['CCDC_SR', 'EVHR_TOA', 'EVHR_SRLite', 'EVHR_SR_RMA'],
                                              i='index', j='Band', suffix='\D+').reset_index()

            from pandas.api.types import CategoricalDtype
            bandsType = CategoricalDtype(categories=['Band-B'], ordered=True)
    #        bandsType = CategoricalDtype(categories=['Blue', 'Green', 'Red', 'NIR'], ordered=True)
            reflect_df_long['Band'] = reflect_df_long['Band'].astype(bandsType)

            #       def _sr_performance(self, context, df, sr_model, bandName):

            metrics_srlite = pd.concat([
                 pd.DataFrame([self._sr_performance(context, reflect_df_long, model_data_only_band, 'Band-B')])
            ]).reset_index()

        return sr_prediction_band, metrics_srlite

    def mean_bias_error(self, y_true, y_pred):
            '''
            Parameters:
                y_true (array): Array of observed values
                y_pred (array): Array of prediction values

            Returns:
                mbe (float): Bias score
            '''
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_true = y_true.reshape(len(y_true), 1)
            y_pred = y_pred.reshape(len(y_pred), 1)
            diff = (y_true - y_pred)
            mbe = diff.mean()
            # print('MBE = ', mbe)
            return mbe

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

    # reflect_df = pd.concat([
    #     ma2df(warp_ma_masked_list[0], 'CCDC_SR', 'Blue'),
    #     ma2df(warp_ma_masked_list[1], 'CCDC_SR', 'Green'),
    #     ma2df(warp_ma_masked_list[2], 'CCDC_SR', 'Red'),
    #     ma2df(warp_ma_masked_list[3], 'CCDC_SR', 'NIR'),
    #     ma2df(warp_ma_masked_list[4], 'EVHR_TOA', 'Blue'),
    #     ma2df(warp_ma_masked_list[5], 'EVHR_TOA', 'Green'),
    #     ma2df(warp_ma_masked_list[6], 'EVHR_TOA', 'Red'),
    #     ma2df(warp_ma_masked_list[7], 'EVHR_TOA', 'NIR'),
    #     ma2df(warp_ma_masked_list[8], 'EVHR_SRLite', 'Blue'),
    #     ma2df(warp_ma_masked_list[9], 'EVHR_SRLite', 'Green'),
    #     ma2df(warp_ma_masked_list[10], 'EVHR_SRLite', 'Red'),
    #     ma2df(warp_ma_masked_list[11], 'EVHR_SRLite', 'NIR')],
    #     axis=1)

    def _sr_performance(self, context, df, sr_model, bandName):

            sr = df[df['Band'] == bandName]
#            sr_model = LinearRegression().fit(sr['EVHR_SRLite'].values.reshape(-1, 1), sr['CCDC_SR'])
            intercept = sr_model.intercept_
            slope = sr_model.coef_
            # score = sr_model.score(sr['EVHR_SRLite'].values.reshape(-1, 1), sr['CCDC_SR'])
            # r2_score = sklearn.metrics.r2_score(sr['CCDC_SR'].values.reshape(-1, 1), sr['EVHR_SRLite'])
            # explained_variance = sklearn.metrics.explained_variance_score(sr['CCDC_SR'].values.reshape(-1, 1),
            #                                                               sr['EVHR_SRLite'])
            # mbe = self.mean_bias_error(sr['CCDC_SR'].values.reshape(-1, 1), sr['EVHR_SRLite'])
            # mae = sklearn.metrics.mean_absolute_error(sr['CCDC_SR'].values.reshape(-1, 1), sr['EVHR_SRLite'])
            # mape = sklearn.metrics.mean_absolute_percentage_error(sr['CCDC_SR'].values.reshape(-1, 1),
            #                                                       sr['EVHR_SRLite'])
            # medae = sklearn.metrics.median_absolute_error(sr['CCDC_SR'].values.reshape(-1, 1), sr['EVHR_SRLite'])
            # mse = sklearn.metrics.mean_squared_error(sr['CCDC_SR'].values.reshape(-1, 1), sr['EVHR_SRLite'])
            # rmse = mse ** 0.5
            # mean_ccdc_sr = sr['CCDC_SR'].mean()
            # mean_evhr_srlite = sr['EVHR_SRLite'].mean()
            # mae_norm = mae / mean_ccdc_sr
            # rmse_norm = rmse / mean_ccdc_sr

            metadata = {'band': bandName,
                        'intercept': intercept, 'slope': slope[0]}

            # metadata =  {'region': context['region'], 'scene': context['scene'], 'band': bandName, 'version': context['version'], \
            #         'intercept': intercept, 'slope': slope[0], 'score': score, 'r2_score': r2_score,
            #         'explained_variance': explained_variance, \
            #         'mbe': mbe, 'mae': mae, 'mape': mape, 'medae': medae, 'rmse': rmse, \
            #         'mean_ccdc_sr': mean_ccdc_sr, 'mean_evhr_srlite': mean_evhr_srlite, 'mae_norm': mae_norm,
            #         'rmse_norm': rmse_norm}

            return metadata

    def simulateSurfaceReflectance(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]
#        numBandPairs = len(bandPairIndicesList)

        sr_prediction_list = []
        warp_ds_list = context[Context.DS_WARP_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        minWarning = 0

        self.prepareMasks(context)

        ########################################
        # ### FOR EACH BAND PAIR,
        # now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
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
            common_mask_band_all = self._getCommonMask(context, targetBandMaArray, toaBandMaArray)

            # Apply the 3-way common mask to the CCDC and EVHR bands
            warp_ma_masked_band_list = [np.ma.array(targetBandMaArray, mask=common_mask_band_all),
                                        np.ma.array(toaBandMaArray, mask=common_mask_band_all)]

             # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
#                    exit(1)

            ########################################
            # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
            # CCDC SR is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
            ########################################

            # Get 2m TOA Masked Array
            toaBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA], bandPairIndices[context[Context.LIST_INDEX_TOA]])
            sr_prediction_band, metadata = self.predictSurfaceReflectance(context,
                                                                "BAND-B",
                                                                toaBandMaArrayRaw,
                                                                warp_ma_masked_band_list[context[Context.LIST_INDEX_TARGET]],
                                                                warp_ma_masked_band_list[context[Context.LIST_INDEX_TOA]])

            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file '
                f'{os.path.basename(context[Context.FN_LIST][context[Context.LIST_INDEX_TOA]])}')

            # Return to original shape and apply original mask
            toa_sr_ma_band = np.ma.array(sr_prediction_band.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask)

            # Check resulting ma
            self._plot_lib.trace(f'Input masked array shape: {toaBandMaArray.shape} and Final masked array shape: {toa_sr_ma_band.shape}')
            self._plot_lib.trace(f'Metrics: {metadata}')

            ########### save prediction for each band #############
            sr_prediction_list.append(toa_sr_ma_band)

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
                dst.set_band_description(id+1, str(band_description_list[id]))
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

