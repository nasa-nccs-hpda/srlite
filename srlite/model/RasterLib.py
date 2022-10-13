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
# Not in current ilab kernel (or introduced from diagnostics
from pylr2 import regress2
from sklearn.linear_model import HuberRegressor, LinearRegression
from pathlib import Path
from srlite.model.Context import Context


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
            print('ERROR - check gdal version: ', err)
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

    def b_getma(self, b):
        """Get masked array from input GDAL Band

        Parameters
        ----------
        b : gdal.Band
            Input GDAL Band

        Returns
        -------
        np.ma.array
            Masked array containing raster values
        """
        b_ndv = iolib.get_ndv_b(b)
        #bma = np.ma.masked_equal(b.ReadAsArray(), b_ndv)
        #This is more appropriate for float, handles precision issues
        bma = np.ma.masked_values(b.ReadAsArray(), b_ndv, shrink=False)
        return bma

    #Given input dataset, return a masked array for the input band
    def ds_getma(self, ds, bnum=1):
        """Get masked array from input GDAL Dataset

        Parameters
        ----------
        ds : gdal.Dataset
            Input GDAL Datset
        bnum : int, optional
            Band number

        Returns
        -------
        np.ma.array
            Masked array containing raster values
        """
        b = ds.GetRasterBand(bnum)
        return b_getma(b)

    #Masked Array to 1D
    def ma2_1d(self, ma):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False]
        return np.array(unmasked)

    #Masked Array to data frame
    def ma2df(self, ma, product, band):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False]
        df = pd.DataFrame(unmasked)
        df.columns = [product + band]
        df[product + band] = df[product + band] * 0.0001
        return df#Given input band, return a masked array

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
        # Extract old ndv
        old_ndv = iolib.get_ndv_b(b)

        print(src_fn)
        print("Replacing old ndv %s with new ndv %s" % (old_ndv, new_ndv))

        # Load masked array
        bma = iolib.ds_getma(ds)

        # Handle cases with input ndv of nan
        # if old_ndv == np.nan:
        bma = np.ma.fix_invalid(bma)

        # Set new fill value
        bma.set_fill_value(new_ndv)
        # Fill ma with new value and write out
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
                self._plot_lib.trace(
                    f'Origin: ({geotransform[0]}, '
                    f'{geotransform[3]}), Resolution: ({geotransform[1]}, {geotransform[5]})  ')
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
            context[Context.DS_INTERSECTION_LIST], res='first', extent='intersection', t_srs='first',
            r=context[Context.TARGET_SAMPLING_METHOD])
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        self._plot_lib.trace(
            '\n TOA shape=' + str(warp_ma_list[0].shape) + ' TARGET shape=' + str(warp_ma_list[1].shape))
        return warp_ds_list, warp_ma_list

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
        context[Context.MA_LIST] = [iolib.fn_getma(fn) for fn in context[Context.FN_LIST]]
        return context[Context.MA_LIST]

    def maskNegativeValues(self, context):
        self._validateParms(context, [Context.MA_LIST])

        warp_valid_ma_list = [np.ma.masked_where(ma < 0, ma) for ma in context[Context.MA_LIST]]
        return warp_valid_ma_list

    def getIntersection(self, context):
        self._validateParms(context, [Context.FN_INTERSECTION_LIST])

        # ########################################
        # # Take the intersection of the scenes and return masked arrays of common pixels
        # ########################################
        # warp_ds_list = warplib.memwarp_multi_fn(
        warp_ds_list = warplib.memwarp_multi_fn(
            context[Context.FN_INTERSECTION_LIST], res='first', extent='intersection', t_srs='first',
            r=context[Context.TARGET_SAMPLING_METHOD])
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        return warp_ds_list, warp_ma_list

    def getCcdcReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST, Context.FN_TARGET, Context.FN_TOA])

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

        return warp_ds_list, warp_ma_list

    def getReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST, Context.FN_REPROJECTION_LIST, Context.TARGET_FN,
                                      Context.TARGET_SAMPLING_METHOD])

        # ########################################
        # # Align context[Context.FN_LIST[>0]] to context[Context.FN_LIST[0]] return masked arrays of reprojected pixels
        # ########################################
        ndv_list = [self.get_ndv(fn) for fn in context[Context.FN_REPROJECTION_LIST]]
#        self._plot_lib.trace(f'Fill values before re-projection:  {ndv_list}')

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
#        self._plot_lib.trace(f'Fill values after re-projection:  {[ma.get_fill_value() for ma in warp_ma_list]}')

        return warp_ds_list, warp_ma_list

    def __getReprojection(self, context):
        self._validateParms(context, [Context.FN_LIST])

        # ########################################
        # # Align context[Context.FN_LIST[>0]] to context[Context.FN_LIST[0]] return masked arrays of reprojected pixels
        # ########################################
        warp_ds_list = warplib.memwarp_multi_fn(
            context[Context.FN_LIST], res=context[Context.TARGET_XRES], extent='first', t_srs='first')
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
            np.ma.masked_where(cloudmask_warp_ma >= 0.5, cloudmask_warp_ma)
#        np.ma.masked_where(cloudmask_warp_ma == 1.0, cloudmask_warp_ma)

        return cloudmaskWarpExternalBandMaArrayMasked

    def _prepareEVHRCloudmask(self, context):
        self._validateParms(context,
                            [Context.MA_CLOUDMASK_DOWNSCALE])

        cloudmaskWarpExternalBandMaArray = context[Context.MA_CLOUDMASK_DOWNSCALE]
        cloudmaskWarpExternalBandMaArraycloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(
            cloudmaskWarpExternalBandMaArray == 1.0,
            cloudmaskWarpExternalBandMaArray)
        return cloudmaskWarpExternalBandMaArraycloudmaskWarpExternalBandMaArrayMasked

    def __prepareEVHRCloudmask(self, context):
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
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArraynp.ma.count (non-masked)=' + str(count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArraynp.ma.count_masked (masked)=' + str(count_masked))
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
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked shape: {cloudmaskWarpExternalBandMaArrayMasked.shape}')
        count_non_masked = np.ma.count(cloudmaskWarpExternalBandMaArrayMasked)
        count_masked = np.ma.count_masked(cloudmaskWarpExternalBandMaArrayMasked)
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMaskednp.ma.count (masked)=' + str(count_non_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMaskednp.ma.count_masked (non-masked)=' + str(count_masked))
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
                            [Context.MA_WARP_LIST, Context.LIST_INDEX_CLOUDMASK])

        # Mask out clouds
        cloudmask_warp_ds_target = context[Context.DS_WARP_LIST][context[Context.LIST_INDEX_TARGET]]
        cloudmask_warp_ma = iolib.ds_getma(cloudmask_warp_ds_target, 8)
        # cloudmaskWarpExternalBandMaArray = iolib.fn_getma(context[Context.FN_TARGET_DOWNSCALE], 8)
        # cloudmaskWarpExternalBandMaArrayMasked = \
        #     np.ma.masked_where(cloudmask_warp_ma == 1.0, cloudmask_warp_ma)

        # Create a mask where the pixel values equal to '0, 3, 4' are suppressed because these correspond to NoData,
        # Clouds, and Cloud Shadows
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

    def _prepareQualityFlagMask(self, context):
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
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArraynp.ma.count (masked)=' + str(count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArraynp.ma.count_masked (non-masked)=' + str(count_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArray total count (masked + non-masked)=' + str(
                count_masked + count_non_masked))
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArray max=' + str(cloudmaskWarpExternalBandMaArray.max()))
        self._plot_lib.plot_combo_array(cloudmaskWarpExternalBandMaArray, figsize=(14, 7),
                                        title='cloudmaskWarpExternalBandMaArray')

        # Create a mask where the pixel values equal to '0, 3, 4' are suppressed because these
        # correspond to NoData, Clouds, and Cloud Shadows
        self._plot_lib.trace(
            f'\nSuppress values=[0, 3, 4] according to Band #8 because they correspond to NoData, Clouds, '
            f'and Cloud Shadows')

        ndv = int(Context.DEFAULT_NODATA_VALUE)
        cloudmaskWarpExternalBandMaArrayData = np.ma.getdata(cloudmaskWarpExternalBandMaArray)
        cloudmaskWarpExternalBandMaArrayData = np.select(
            [cloudmaskWarpExternalBandMaArrayData == 0, cloudmaskWarpExternalBandMaArrayData == 3,
             cloudmaskWarpExternalBandMaArrayData == 4], [ndv, ndv, ndv], cloudmaskWarpExternalBandMaArrayData)
        cloudmaskWarpExternalBandMaArrayData = np.select([cloudmaskWarpExternalBandMaArrayData != ndv], [0.0],
                                                         cloudmaskWarpExternalBandMaArrayData)
        cloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(cloudmaskWarpExternalBandMaArrayData == ndv,
                                                                    cloudmaskWarpExternalBandMaArrayData)

        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked hist: {np.histogram(cloudmaskWarpExternalBandMaArrayMasked)}')
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMasked shape: {cloudmaskWarpExternalBandMaArrayMasked.shape}')
        count_non_masked = np.ma.count(cloudmaskWarpExternalBandMaArrayMasked)
        count_masked = np.ma.count_masked(cloudmaskWarpExternalBandMaArrayMasked)
        self._plot_lib.trace(f'cloudmaskWarpExternalBandMaArrayMaskednp.ma.count (masked)=' + str(count_non_masked))
        self._plot_lib.trace(
            f'cloudmaskWarpExternalBandMaArrayMaskednp.ma.count_masked (non-masked)=' + str(count_masked))
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
        return df

    def generateCSV(self, context, sr_metrics_list):
        if (eval(context[Context.CSV_FLAG])):
            if (context[Context.BATCH_NAME] != 'None'):
                figureBase = context[Context.BATCH_NAME] + '_' + context[Context.FN_PREFIX] \
                             + '_' + context[Context.REGRESSION_MODEL]
            else:
                figureBase = context[Context.FN_PREFIX] \
                             + '_' + context[Context.REGRESSION_MODEL]
            path = os.path.join(context[Context.DIR_OUTPUT_CSV],
                                figureBase + '_SRLite_statistics.csv')
            sr_metrics_list.to_csv(path)
            self._plot_lib.trace(
                f"\nCreated CSV with coefficients for batch {context[Context.BATCH_NAME]}...\n   {path}")

    def prepareMasks(self, context):

        # Get optional Cloudmask
        if (eval(context[Context.CLOUD_MASK_FLAG])):
            context['cloudmaskEVHRWarpExternalBandMaArrayMasked'] = self.prepareEVHRCloudmask(context)

        # Get optional Quality flag mask
        if (eval(context[Context.QUALITY_MASK_FLAG])):
            context['cloudmaskQFWarpExternalBandMaArrayMasked'] = self.prepareQualityFlagMask(context)

    def _getCommonMask(self, context, targetBandArray, toaBandArray):

        context['evhrBandMaThresholdArray'] = None
        #  Create a common mask that intersects the CCDC/QF, EVHR, and Cloudmasks
        #  - this will then be used to correct the input EVHR & CCDC/QF
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

    #Masked Array to 1D
    def ma2_1d(self, ma):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False]
        return np.array(unmasked)

    #Masked Array to data frame
    def ma2df(self, ma, product, band):
        raveled = ma.ravel()
        unmasked = raveled[raveled.mask == False]
        df = pd.DataFrame(unmasked)
        df.columns = [product + band]
        df[product + band] = df[product + band] * 0.0001
        return df

    #Given input dataset, return a masked array for the input band
    def ds_getma(self, ds, bnum=1):
        """Get masked array from input GDAL Dataset

        Parameters
        ----------
        ds : gdal.Dataset
            Input GDAL Datset
        bnum : int, optional
            Band number

        Returns
        -------
        np.ma.array
            Masked array containing raster values
        """
        b = ds.GetRasterBand(bnum)
        return self.b_getma(b)

    #Given input band, return a masked array
    def b_getma(self, b):
        """Get masked array from input GDAL Band

        Parameters
        ----------
        b : gdal.Band
            Input GDAL Band

        Returns
        -------
        np.ma.array
            Masked array containing raster values
        """
        b_ndv = iolib.get_ndv_b(b)
        #bma = np.ma.masked_equal(b.ReadAsArray(), b_ndv)
        #This is more appropriate for float, handles precision issues
        bma = np.ma.masked_values(b.ReadAsArray(), b_ndv, shrink=False)
        return bma

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
        y_true = y_true.reshape(len(y_true),1)
        y_pred = y_pred.reshape(len(y_pred),1)
        diff = (y_true-y_pred)
        mbe = diff.mean()
        return mbe

    def sr_performance(self, df, bandName, model, intercept, slope, ndv_value=None):

        metadata = {}
        metadata['band_name'] = bandName
        metadata['model'] = model
        metadata['intercept'] = intercept
        metadata['slope'] = slope

        if (ndv_value == None):
            sr = df[df['Band'] == bandName]
            metadata['r2_score'] = sklearn.metrics.r2_score(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['explained_variance'] = sklearn.metrics.explained_variance_score(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['mbe'] = self.mean_bias_error(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['mae'] = sklearn.metrics.mean_absolute_error(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['mape'] = sklearn.metrics.mean_absolute_percentage_error(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['medae'] = sklearn.metrics.median_absolute_error(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['mse'] = sklearn.metrics.mean_squared_error(sr['CCDC_SR'].values.reshape(-1,1), sr['EVHR_SRLite'])
            metadata['rmse'] = metadata['mse'] ** 0.5
            metadata['mean_ccdc_sr'] = sr['CCDC_SR'].mean()
            metadata['mean_evhr_srlite'] = sr['EVHR_SRLite'].mean()
            metadata['mae_norm'] = metadata['mae'] / metadata['mean_ccdc_sr']
            metadata['rmse_norm'] =  metadata['rmse'] / metadata['mean_ccdc_sr']
        else:
            metadata['r2_score'] = ndv_value
            metadata['explained_variance'] = ndv_value
            metadata['mbe'] = ndv_value
            metadata['mae'] = ndv_value
            metadata['mape'] = ndv_value
            metadata['medae'] = ndv_value
            metadata['mse'] = ndv_value
            metadata['rmse'] = ndv_value
            metadata['mean_ccdc_sr'] = ndv_value
            metadata['mean_evhr_srlite'] = ndv_value
            metadata['mae_norm'] = ndv_value
            metadata['rmse_norm'] = ndv_value

        return metadata

    def apply_regressor_otf_8band(self, context, image_4band):

        # Get coefficients for standard 4-Bands
        sr_metrics_list = context[Context.METRICS_LIST]
        model = context[Context.REGRESSION_MODEL]

        # Correction coefficients for simulated bands
        yellowGreenCorr = 0.473
        yellowRedCorr = 0.527
        rededgeRedCorr = 0.621
        rededgeNIR1Corr = 0.379

        # Retrieve slope, intercept, and score coefficients for data-driven bands
        blueSlope = sr_metrics_list['slope'][0]
        blueIntercept = sr_metrics_list['intercept'][0]

        greenSlope = sr_metrics_list['slope'][1]
        greenIntercept = sr_metrics_list['intercept'][1]

        redSlope = sr_metrics_list['slope'][2]
        redIntercept = sr_metrics_list['intercept'][2]

        NIR1Slope = sr_metrics_list['slope'][3]
        NIR1Intercept = sr_metrics_list['intercept'][3]

        # Read CCDC, SRLite, and Cloud images
        ccdcImage = os.path.join(context[Context.FN_TARGET])
        evhrSrliteImage = os.path.join(context[Context.FN_COG])
        cloudImage = os.path.join(context[Context.FN_CLOUDMASK])

        _ndv=-9999
        fn_list = [ccdcImage, evhrSrliteImage, cloudImage]
        warp_ds_list = warplib.memwarp_multi_fn(fn_list, res=30, extent=evhrSrliteImage,
                                                t_srs=evhrSrliteImage, r='average', dst_ndv=_ndv)

        # get similar bands across arrays
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

        # Pull out flat masked arrays
        ccdc_blu = self.ma2_1d(warp_ma_masked_list[0])
        ccdc_grn = self.ma2_1d(warp_ma_masked_list[1])
        ccdc_red = self.ma2_1d(warp_ma_masked_list[2])
        ccdc_nir = self.ma2_1d(warp_ma_masked_list[3])

        evhr_srlite_blu = self.ma2_1d(warp_ma_masked_list[4])
        evhr_srlite_grn = self.ma2_1d(warp_ma_masked_list[5])
        evhr_srlite_red = self.ma2_1d(warp_ma_masked_list[6])
        evhr_srlite_nir = self.ma2_1d(warp_ma_masked_list[7])

        # Create a dataframw with the arrays
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
#        bandsType = CategoricalDtype(categories = ['BAND-B','BAND-G','BAND-R','BAND-N'], ordered=True)
        bandsType = CategoricalDtype(categories = ['Blue','Green','Red','NIR'], ordered=True)
        reflect_df_long['Band'] = reflect_df_long['Band'].astype(bandsType)

        # Grab the Blue band from the dataframe
        _sr = reflect_df_long[reflect_df_long['Band'] == 'Blue']
        debug = '0'
        if (debug == '1'):
            print(sr)
            print("\nsr['CCDC_SR']\n", sr['CCDC_SR'])
            print("\nsr['CCDC_SR'].values.reshape(-1,1)\n", sr['CCDC_SR'].values.reshape(-1,1))
            print("\nsr['EVHR_SRLite']\n", sr['EVHR_SRLite'])
            print("\nsr['EVHR_SRLite'].values.reshape(-1,1)\n", sr['EVHR_SRLite'].values.reshape(-1,1))

        r2_score = sklearn.metrics.r2_score(_sr['CCDC_SR'].values.reshape(-1,1), _sr['EVHR_SRLite'])

        # Create table of dataframes with coefficients & statistics
        ndv_value = "NA"
        metrics_srlite_long = pd.concat([
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'Coastal', model, blueIntercept, blueSlope, ndv_value)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'Blue', model, blueIntercept, blueSlope)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'Green', model, greenIntercept, greenSlope)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'Yellow', model,
                                              (greenIntercept * yellowGreenCorr) + (redIntercept * yellowRedCorr),
                                              (greenSlope * yellowGreenCorr) + (redSlope * yellowRedCorr),
                                              ndv_value)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'Red', model, redIntercept, redSlope)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'RedEdge', model,
                                              (redIntercept * rededgeRedCorr) + (NIR1Intercept * rededgeNIR1Corr),
                                              (redSlope * rededgeRedCorr) + (NIR1Slope * rededgeNIR1Corr),
                                              ndv_value)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'NIR', model, NIR1Intercept, NIR1Slope)]),
            pd.DataFrame([self.sr_performance(reflect_df_long,
                                              'NIR2', model, NIR1Intercept, NIR1Slope, ndv_value)]),
        ]).reset_index()

        # # Now that CSV is taken care of, create/update bands
        # evhrToa8 = [context[Context.FN_TOA]]
        # evhrToa = gdal.Open(str(context[Context.FN_TOA]))
        # warp_ds_list_toa8 = warplib.memwarp_multi_fn(
        #     evhrToa8, res=evhrToa, extent=evhrToa, t_srs=evhrToa, r='average', dst_ndv=_ndv)

        # warp_ma_list_toa8_ = [self.ds_getma(warp_ds_list_toa8[0],1), self.ds_getma(warp_ds_list_toa8[0],2),
        #                      self.ds_getma(warp_ds_list_toa8[0],3), self.ds_getma(warp_ds_list_toa8[0],4),
        #                      self.ds_getma(warp_ds_list_toa8[0],5), self.ds_getma(warp_ds_list_toa8[0],6),
        #                      self.ds_getma(warp_ds_list_toa8[0],7), self.ds_getma(warp_ds_list_toa8[0],8)]
        #
        # warp_ma_list_toa8 = [iolib.fn_getma(context[Context.FN_TOA],1), iolib.fn_getma(context[Context.FN_TOA],2),
        #                      iolib.fn_getma(context[Context.FN_TOA],3), iolib.fn_getma(context[Context.FN_TOA],4),
        #                      iolib.fn_getma(context[Context.FN_TOA],5), iolib.fn_getma(context[Context.FN_TOA],6),
        #                      iolib.fn_getma(context[Context.FN_TOA],7), iolib.fn_getma(context[Context.FN_TOA],8)]

        toa_ma_band_coastal = iolib.fn_getma(context[Context.FN_TOA],1)
        toa_ma_band_blue = iolib.fn_getma(context[Context.FN_TOA],2)
        toa_ma_band_green = iolib.fn_getma(context[Context.FN_TOA],3)
        toa_ma_band_yellow = iolib.fn_getma(context[Context.FN_TOA],4)
        toa_ma_band_red = iolib.fn_getma(context[Context.FN_TOA],5)
        toa_ma_band_rededge = iolib.fn_getma(context[Context.FN_TOA],6)
        toa_ma_band_nir1 = iolib.fn_getma(context[Context.FN_TOA],7)
        toa_ma_band_nir2 = iolib.fn_getma(context[Context.FN_TOA],8)

        sr_prediction_band_coastal = \
            ((toa_ma_band_coastal * metrics_srlite_long['slope'][0]) + (metrics_srlite_long['intercept'][0] * 10000))
        sr_prediction_band_reshaped_coastal = sr_prediction_band_coastal.reshape(toa_ma_band_coastal.shape)
        sr_prediction_band_ma_coastal = np.ma.array(
            sr_prediction_band_reshaped_coastal,
            mask=toa_ma_band_coastal.mask)

        sr_prediction_band_blue = \
            ((toa_ma_band_blue * metrics_srlite_long['slope'][0]) + (metrics_srlite_long['intercept'][0] * 10000))
        sr_prediction_band_reshaped_blue = sr_prediction_band_blue.reshape(toa_ma_band_blue.shape)
        sr_prediction_band_ma_blue = np.ma.array(
            sr_prediction_band_reshaped_blue,
            mask=toa_ma_band_blue.mask)

        sr_prediction_band_green = \
            ((toa_ma_band_green * metrics_srlite_long['slope'][1]) + (metrics_srlite_long['intercept'][1] * 10000))
        sr_prediction_band_reshaped_green = sr_prediction_band_green.reshape(toa_ma_band_green.shape)
        sr_prediction_band_ma_green = np.ma.array(
            sr_prediction_band_reshaped_green,
            mask=toa_ma_band_green.mask)

        sr_prediction_band_yellow = \
            (((toa_ma_band_yellow.astype(float) * metrics_srlite_long['slope'][1]) + (metrics_srlite_long['intercept'][1] * 10000)) * yellowGreenCorr) + (((toa_ma_band_yellow.astype(float) * metrics_srlite_long['slope'][2]) + (metrics_srlite_long['intercept'][2] * 10000)) * yellowRedCorr)
        sr_prediction_band_reshaped_yellow = sr_prediction_band_yellow.reshape(toa_ma_band_yellow.shape)
        sr_prediction_band_ma_yellow = np.ma.array(
            sr_prediction_band_reshaped_yellow,
            mask=toa_ma_band_yellow.mask)

        sr_prediction_band_red = \
            ((toa_ma_band_red * metrics_srlite_long['slope'][2]) + (metrics_srlite_long['intercept'][2] * 10000))
        sr_prediction_band_reshaped_red = sr_prediction_band_red.reshape(toa_ma_band_red.shape)
        sr_prediction_band_ma_red = np.ma.array(
            sr_prediction_band_reshaped_red,
            mask=toa_ma_band_red.mask)

        sr_prediction_band_rededge = \
            (((toa_ma_band_rededge.astype(float) * metrics_srlite_long['slope'][2]) + (metrics_srlite_long['intercept'][2] * 10000)) * rededgeRedCorr) + (((toa_ma_band_rededge.astype(float) * metrics_srlite_long['slope'][3]) + (metrics_srlite_long['intercept'][3] * 10000)) * rededgeNIR1Corr)
        sr_prediction_band_reshaped_rededge = sr_prediction_band_rededge.reshape(toa_ma_band_rededge.shape)
        sr_prediction_band_ma_rededge = np.ma.array(
            sr_prediction_band_reshaped_rededge,
            mask=toa_ma_band_rededge.mask)

        sr_prediction_band_nir1 = \
            ((toa_ma_band_nir1 * metrics_srlite_long['slope'][3]) + (metrics_srlite_long['intercept'][3] * 10000))
        sr_prediction_band_reshaped_nir1 = sr_prediction_band_nir1.reshape(toa_ma_band_nir1.shape)
        sr_prediction_band_ma_nir1 = np.ma.array(
            sr_prediction_band_reshaped_nir1,
            mask=toa_ma_band_nir1.mask)

        sr_prediction_band_nir2 = \
            ((toa_ma_band_nir2 * metrics_srlite_long['slope'][3]) + (metrics_srlite_long['intercept'][3] * 10000))
        sr_prediction_band_reshaped_nir2 = sr_prediction_band_nir2.reshape(toa_ma_band_nir2.shape)
        sr_prediction_band_ma_nir2 = np.ma.array(
            sr_prediction_band_reshaped_nir2,
            mask=toa_ma_band_nir2.mask)

        result_weighted_8band = [
            sr_prediction_band_ma_coastal, sr_prediction_band_ma_blue, sr_prediction_band_ma_green, sr_prediction_band_ma_yellow,
            sr_prediction_band_ma_red, sr_prediction_band_ma_rededge, sr_prediction_band_ma_nir1, sr_prediction_band_ma_nir2
        ]

        return context[Context.FN_COG], metrics_srlite_long, result_weighted_8band

    def _apply_regressor_otf_8band(self, context, image_4band, sr_unmasked_prediction_list,
                                  sr_metrics_list):

         # Retrieve slope, intercept, and score coefficients for data-driven bands
        blueSlope = sr_metrics_list['slope'][0]
        blueIntercept = sr_metrics_list['intercept'][0]
        blueTargetBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TARGET],1)
        blueSrliteBandMaArrayRaw = iolib.fn_getma(context[Context.FN_COG],1)

        greenSlope = sr_metrics_list['slope'][1]
        greenIntercept = sr_metrics_list['intercept'][1]
        greenTargetBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TARGET],2)
        greenSrliteBandMaArrayRaw = iolib.fn_getma(context[Context.FN_COG],2)

        redSlope = sr_metrics_list['slope'][2]
        redIntercept = sr_metrics_list['intercept'][2]
        redTargetBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TARGET],3)
        redSrliteBandMaArrayRaw = iolib.fn_getma(context[Context.FN_COG],3)

        NIR1Slope = sr_metrics_list['slope'][3]
        NIR1Intercept = sr_metrics_list['intercept'][3]
        NIR1TargetBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TARGET],3)
        NIR1SrliteBandMaArrayRaw = iolib.fn_getma(context[Context.FN_COG],3)

        # Simulated bands - coefficients provided via Matt
        #    docs.google.com/presentation/d/1zqnpiIADrUttMnWsr4M3lA8XChQWUygEQLlgp1GW9vQ/edit#slide=id.gfde680b449_0_0

        # Correction coefficients for simulated bands
        yellowGreenCorr = 0.473
        yellowRedCorr = 0.527
        rededgeRedCorr = 0.621
        RededgeNIR1Corr = 0.379

        # Generate diagnostics for simulated bands - use fill value to indicate bogus data
        bandAerosolMetadata = self._model_coeffs_(
            context,
            'BAND-C',
            context[Context.REGRESSION_MODEL],
            blueIntercept,
            blueSlope,
            None,
            None,
            ndv_value='NA'
        )
        bandYellowMetadata = self._model_coeffs_(
            context,
            'BAND-Y',
            context[Context.REGRESSION_MODEL],
            # (((maListToa[3].astype(float) * params_df.query('band == "Green"')['slope'].reset_index(drop=True)[0])
            # + (params_df.query('band == "Green"')['intercept'].reset_index(drop=True)[0] * 10000)) * 0.518) +
            # (((maListToa[3].astype(float) * params_df.query('band == "Red"')['slope'].reset_index(drop=True)[0])
            # + (params_df.query('band == "Red"')['intercept'].reset_index(drop=True)[0] * 10000)) * 0.482),
            (greenIntercept * yellowGreenCorr) + (redIntercept * yellowRedCorr),
            (greenSlope * yellowGreenCorr) + (redSlope * yellowRedCorr),
            None,
            None,
            ndv_value='NA'
        )
        bandRededgeMetadata = self._model_coeffs_(
            # (((maListToa[5].astype(float) * params_df.query('band == "Red"')['slope'].reset_index(drop=True)[0])
            # +(params_df.query('band == "Red"')['intercept'].reset_index(drop=True)[0] * 10000)) * 0.379) +
            # (((maListToa[5].astype(float) * params_df.query('band == "NIR"')['slope'].reset_index(drop=True)[0])
            # + (params_df.query('band == "NIR"')['intercept'].reset_index(drop=True)[0] * 10000)) * 0.621),
            context,
            'BAND-RE',
            context[Context.REGRESSION_MODEL],
            (redIntercept * rededgeRedCorr) + (NIR1Intercept * RededgeNIR1Corr),
            (redSlope * rededgeRedCorr) + (NIR1Slope * RededgeNIR1Corr),
            None,
            None,
            ndv_value='NA'
        )
        bandNIR2Metadata = self._model_coeffs_(
            context,
            'BAND-N2',
            context[Context.REGRESSION_MODEL],
            NIR1Intercept,
            NIR1Slope,
            None,
            None,
            ndv_value='NA'
        )
        sr_metrics_list = pd.concat([sr_metrics_list, pd.DataFrame([bandAerosolMetadata], index=[4])])
        sr_metrics_list = pd.concat([sr_metrics_list, pd.DataFrame([bandYellowMetadata], index=[5])])
        sr_metrics_list = pd.concat([sr_metrics_list, pd.DataFrame([bandRededgeMetadata], index=[6])])
        sr_metrics_list = pd.concat([sr_metrics_list, pd.DataFrame([bandNIR2Metadata], index=[7])])
        sr_metrics_list.reset_index()

        # Generate weighted bands
        bandAerosol = bandBlue
        bandYellow = (bandGreen * yellowGreenCorr) + (bandRed * yellowRedCorr)
        bandRededge = (bandRed * rededgeRedCorr) + (bandNIR1 * RededgeNIR1Corr)
        bandNIR2 = bandNIR1

#        sr_prediction_band = (toa_hr_band * metadata['slope']) + (metadata['intercept'] * 10000)
        result_weighted_8band = [
            # Aerosol
            np.ma.array(bandAerosol.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # Blue
            np.ma.array(bandBlue.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # Green
            np.ma.array(bandGreen.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # Yellow
            np.ma.array(bandYellow.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # Red
            np.ma.array(bandRed.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # Rededge
            np.ma.array(bandRededge.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # NIR1
            np.ma.array(bandNIR1.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
            # NIR2
            np.ma.array(bandNIR2.reshape(toaBandMaArrayRaw.shape), mask=toaBandMaArrayRaw.mask),
        ]

        return result_weighted_8band, sr_metrics_list

    def predictSurfaceReflectance(self, context, band_name, toaBandMaArrayRaw,
                                  target_warp_ma_masked_band, toa_warp_ma_masked_band):

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

            # Calculate SR-Lite band using original TOA 2m band
            sr_prediction_band_2m = (toaBandMaArrayRaw * model_data_only_band['slope']) + \
                                 (model_data_only_band['intercept'] * 10000)

            #  band-specific metadata
            metadata = self._model_coeffs_(context,
                                            band_name,
                                            model_data_only_band['intercept'],
                                            model_data_only_band['slope'])

        else:
            print('Invalid regressor specified %s' % context[Context.REGRESSION_MODEL])
            sys.exit(1)

        self._plot_lib.trace(f"\nRegressor=[{context[Context.REGRESSION_MODEL]}] "
          f"slope={metadata['slope']} intercept={metadata['intercept']} ]")
#        f"slope={metadata['slope']} intercept={metadata['intercept']} score=[{metadata['score']}]")

        return sr_prediction_band_2m, metadata

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
        return mbe

    def _model_coeffs_(self, context, band_name, intercept, slope):

        metadata = {}
        metadata['band_name'] = band_name
        metadata['model'] = context[Context.REGRESSION_MODEL]
        metadata['intercept'] = intercept
        metadata['slope'] = slope

        return metadata

    def _model_metrics_(self, context, band_name, intercept, slope, sr_prediction_band_2m, target_sr_data_only_band,
                       ndv_value=None):

        metadata = {}
        metadata['band_name'] = band_name
        metadata['model'] = context[Context.REGRESSION_MODEL]
        metadata['intercept'] = intercept
        metadata['slope'] = slope

        if (ndv_value == None):
            #            sr_model = LinearRegression().fit(sr_prediction_band_30m.reshape(-1, 1), target_sr_data_only_band)
            #            metadata['score'] = sr_model.score(sr_prediction_band_30m.reshape(-1, 1), target_sr_data_only_band)
            metadata['r2_score'] = sklearn.metrics.r2_score(target_sr_data_only_band.reshape(-1, 1),
                                                            sr_prediction_band_30m)
            metadata['explained_variance'] = sklearn.metrics.explained_variance_score(
                target_sr_data_only_band.reshape(-1, 1),
                sr_prediction_band_30m)
            metadata['mbe'] = self.mean_bias_error(target_sr_data_only_band.reshape(-1, 1), sr_prediction_band_30m)
            metadata['mae'] = sklearn.metrics.mean_absolute_error(target_sr_data_only_band.reshape(-1, 1),
                                                                  sr_prediction_band_30m)
            metadata['mape'] = sklearn.metrics.mean_absolute_percentage_error(target_sr_data_only_band.reshape(-1, 1),
                                                                              sr_prediction_band_30m)
            metadata['medae'] = sklearn.metrics.median_absolute_error(target_sr_data_only_band.reshape(-1, 1),
                                                                      sr_prediction_band_30m)
            metadata['mse'] = sklearn.metrics.mean_squared_error(target_sr_data_only_band.reshape(-1, 1),
                                                                 sr_prediction_band_30m)
            metadata['rmse'] = metadata['mse'] ** 0.5
            metadata['mean_ccdc_sr'] = target_sr_data_only_band.mean()
            metadata['mean_evhr_srlite'] = sr_prediction_band_30m.mean()
            metadata['mae_norm'] = metadata['mae'] / metadata['mean_ccdc_sr']
            metadata['rmse_norm'] = metadata['rmse'] / metadata['mean_ccdc_sr']
        else:
            #            metadata['score'] = ndv_value
            metadata['r2_score'] = ndv_value
            metadata['explained_variance'] = ndv_value
            metadata['mbe'] = ndv_value
            metadata['mae'] = ndv_value
            metadata['mape'] = ndv_value
            metadata['medae'] = ndv_value
            metadata['mse'] = ndv_value
            metadata['rmse'] = ndv_value
            metadata['mean_ccdc_sr'] = ndv_value
            metadata['mean_evhr_srlite'] = ndv_value
            metadata['mae_norm'] = ndv_value
            metadata['rmse_norm'] = ndv_value

        return metadata

    def simulateSurfaceReflectance(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]
        #        numBandPairs = len(bandPairIndicesList)

        sr_prediction_list = []
        sr_unmasked_prediction_list = []
        sr_metrics_list = []
        common_mask_list = []
        warp_ds_list = context[Context.DS_WARP_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        minWarning = 0

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
            context[Context.COMMON_MASK] = self._getCommonMask(context, targetBandMaArray, toaBandMaArray)
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
            toaBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA],
                                               bandPairIndices[context[Context.LIST_INDEX_TOA]])
            sr_prediction_band, metadata = self.predictSurfaceReflectance(context,
                                                                          bandNamePairList[bandPairIndex][1],
                                                                          toaBandMaArrayRaw,
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TARGET]],
                                                                          warp_ma_masked_band_list[
                                                                              context[Context.LIST_INDEX_TOA]])

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

        sr_metrics_list.reset_index()
        return sr_prediction_list, sr_metrics_list, common_mask_list

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
        self._plot_lib.trace(f"\nAppy coefficients to "
                             f"{context[Context.BAND_NUM]}-Band High Res File...\n   "
                             f"{str(context[Context.FN_SRC])}")

        now = datetime.now()  # current date and time

        context[Context.FN_SUFFIX] = str(Context.FN_SRLITE_NONCOG_SUFFIX)
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
                dst.set_band_description(id + 1, str(band_description_list[id]))
                bandPrediction1 = np.ma.masked_values(bandPrediction, context[Context.TARGET_NODATA_VALUE])
                dst.write_band(id + 1, bandPrediction1)

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
        self._plot_lib.trace(
            "[ STATS ] = Minimum={}, Maximum={}, Mean={}, StdDev={}".format(stats[0], stats[1], stats[2], stats[3]))

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
                       dstSRS=context[Context.TARGET_SRS], outputType=context[Context.TARGET_OUTPUT_TYPE],
                       xRes=context[Context.TARGET_XRES], yRes=context[Context.TARGET_YRES], outputBounds=extent)
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
        self._plot_lib.trace(
            '======== Applying threshold algorithm to first EVHR Band (Assume Blue) ========================')
        bandMaThresholdMaxArray = np.ma.masked_where(bandMaArray > max, bandMaArray)
        bandMaThresholdRangeArray = np.ma.masked_where(bandMaThresholdMaxArray < min, bandMaThresholdMaxArray)
        self._plot_lib.trace(' threshold range median =' + str(np.ma.median(bandMaThresholdRangeArray)))
        return bandMaThresholdRangeArray

    def get_ndv(self, r_fn):
        with rasterio.open(r_fn) as src:
            return src.profile['nodata']

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
