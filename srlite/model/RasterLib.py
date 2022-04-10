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
from core.model.SystemCommand import SystemCommand
from srlite.model.Context import Context
from sklearn.linear_model import HuberRegressor, LinearRegression

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
        ccdcDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
        ccdcBands = ccdcDs.RasterCount
        evhrDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)
        evhrBands = evhrDs.RasterCount

#        self._plot_lib.trace('bandNamePair: ' + str(bandNamePairList))
        numBandPairs = len(bandNamePairList)
        bandIndices = [numBandPairs]

        for bandPairIndex in range(0, numBandPairs):

            ccdcBandIndex = evhrBandIndex = -1
            currentBandPair = bandNamePairList[bandPairIndex]

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
                self._plot_lib.trace(f"Invalid band pairs - verify correct name and case {currentBandPair}")
                exit(1)

            bandIndices.append([ccdcIndex, evhrIndex])

        ccdcDs = evhrDs = None
        self._plot_lib.trace('validated bandIndices=' + str(bandIndices))
        return bandIndices

    def getAttributeSnapshot(self, context):
        self._validateParms(context, [Context.FN_TOA, Context.FN_CCDC, Context.FN_CLOUDMASK])

        # Get snapshot of attributes of EVHR, CCDC, and Cloudmask tifs and create plot")
        self.getAttributes(str(context[Context.FN_TOA]), "EVHR Combo Plot")
        self.getAttributes(str(context[Context.FN_CCDC]), "CCDC Combo Plot")
        self.getAttributes(str(context[Context.FN_CLOUDMASK]), "Cloudmask Combo Plot")

    def getAttributes(self, r_fn, title):
        r_ds = iolib.fn_getds(r_fn)
        if (self._debug_level >= 1):
            self._plot_lib.trace("File Name is {}".format(r_fn))
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

    def getIntersection(self, context):
        self._validateParms(context, [Context.FN_LIST])

        # ########################################
        # # Align the CCDC and EVHR images, then take the intersection of the grid points
        # ########################################
        warp_ds_list = warplib.memwarp_multi_fn(
            context[Context.FN_LIST], res='first', extent='intersection', t_srs='first', r='average')
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]

        self._plot_lib.trace('\n CCDC shape=' + str(warp_ma_list[0].shape) + ' EVHR shape=' +
                   str(warp_ma_list[1].shape))
        return warp_ds_list, warp_ma_list

    def performRegression(self, context):
        self._validateParms(context,
                            [Context.DS_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.FN_WARP, Context.REGRESSION_MODEL, Context.FN_LIST])

        warp_ds_list = context[Context.DS_LIST]
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]

        ########################################
        # ### PREPARE CLOUDMASK
        # After retrieving the masked array from the warped cloudmask, further reduce it by suppressing the one ("1") value pixels
        ########################################
        self._plot_lib.trace('bandPairIndicesList: ' + str(bandPairIndicesList))
        numBandPairs = len(bandPairIndicesList)
        warp_ma_masked_band_series = [numBandPairs]
        sr_prediction_list = [numBandPairs]

        #  Get Masked array from warped Cloudmask - assumes only 1 band in mask to be applied to all
        cloudmaskWarpExternalBandMaArray = iolib.fn_getma(context[Context.FN_WARP], 1)
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

        #         # Get Cloud mask (assumes 1 band per scene)
        #     cloudmaskArray = iolib.fn_getma(r_fn_cloudmask, 1)
        # #    cloudmaskArray = iolib.ds_getma(warp_ds_list[2], 1)
        #     self._plot_lib.trace(f'cloudmaskArray array shape: {cloudmaskArray.shape}')
        #     cloudmaskMaArray = np.ma.masked_where(cloudmaskArray == 1, cloudmaskArray)
        #     self._plot_lib.trace(f'cloudmaskMaArray array shape: {cloudmaskMaArray.shape}')

        minWarning = 0
        firstBand = True
        threshold = False
        if (threshold == True):
            # TODO replace hard-coded threshold values with user-specified
            # Apply range of -100 to 200 for Blue Band pixel mask and apply to each band
            evhrBandMaArrayThresholdMin = -100
            #    evhrBandMaArrayThresholdMax = 10000
            evhrBandMaArrayThresholdMax = 2000
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
            ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
            evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

            #  Create single mask for all bands based on Blue-band threshold values
            #  Assumes Blue-band is first indice pair, so collect mask on 1st iteration only.
            if (threshold == True):
                if (firstBand == True):
                    evhrBandMaArray = self._applyThreshold(evhrBandMaArrayThresholdMin, evhrBandMaArrayThresholdMax,
                                                               evhrBandMaArray)
                    firstBand = False

            #  Create a common mask that intersects the CCDC, EVHR, and Cloudmask - this will then be used to correct the input EVHR & CCDC
            warp_ma_band_list_all = [ccdcBandMaArray, evhrBandMaArray, cloudmaskWarpExternalBandMaArrayMasked]
            common_mask_band_all = malib.common_mask(warp_ma_band_list_all)

            # Apply the 3-way common mask to the CCDC and EVHR bands
            warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band_all),
                                        np.ma.array(evhrBandMaArray, mask=common_mask_band_all)]

            # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
            #            exit(1)
            self._plot_lib.plot_maps(warp_ma_masked_band_list, context[Context.FN_LIST], figsize=(10, 5),
                              title=str(bandNamePairList[bandPairIndex]) + ' Reflectance (%)')
            self._plot_lib.plot_histograms(warp_ma_masked_band_list, context[Context.FN_LIST], figsize=(10, 3),
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

            # Perform regression fit based on model type!
            if (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_ROBUST):
                model_data_only_band = HuberRegressor().fit(evhr_toa_data_only_band.reshape(-1, 1),
                                                            ccdc_sr_data_only_band)
            else:
                model_data_only_band = LinearRegression().fit(evhr_toa_data_only_band.reshape(-1, 1),
                                                              ccdc_sr_data_only_band)

            self._plot_lib.trace(str(bandNamePairList[bandPairIndex]) + '= > intercept: ' + str(
                model_data_only_band.intercept_) + ' slope: ' + str(model_data_only_band.coef_) + ' score: ' +
                          str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1),
                                                         ccdc_sr_data_only_band)))
            self._plot_lib.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                             model_data_only_band.intercept_)

            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file {os.path.basename(context[Context.FN_LIST][1])}')
            self._plot_lib.trace(f'Input masked array shape: {evhrBandMaArray.shape}')

            score = model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
            self._plot_lib.trace(f'R2 score : {score}')

            # Get 2m EVHR Masked Arrays
            evhrBandMaArrayRaw = iolib.fn_getma(context[Context.FN_LIST][1], bandPairIndices[1])
            sr_prediction_band = model_data_only_band.predict(evhrBandMaArrayRaw.ravel().reshape(-1, 1))
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

            self._plot_lib.plot_histograms(evhr_pre_post_ma_list, context[Context.FN_LIST], figsize=(5, 3),
                                    title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR")
            self._plot_lib.plot_maps(evhr_pre_post_ma_list, compare_name_list, figsize=(10, 50))

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
                                      Context.CLEAN_FLAG, Context.FN_TOA,
                                      Context.LIST_BAND_PAIR_INDICES, Context.FN_SRC,
                                      Context.FN_DEST, Context.FN_WARP,
                                      Context.LIST_BAND_PAIRS, Context.PRED_LIST])

        ########################################
        # Create .tif image from band-based prediction layers
        ########################################
        self._plot_lib.trace(f"\nAppy coefficients to High Res File...{str(context[Context.FN_TOA])}")

        now = datetime.now()  # current date and time

        #  Derive file names for intermediate files
        output_name = "{}/{}".format(
            context[Context.DIR_OUTPUT], str(context[Context.FN_PREFIX])
        ) + "_sr_02m-precog.tif"
        self._plot_lib.trace(f"\nCreating .tif image from band-based prediction layers...{output_name}")

        if eval(context[Context.CLEAN_FLAG]):
            if os.path.exists(output_name):
                os.remove(output_name)

        # Read metadata of EVHR file
        with rasterio.open(str(context[Context.FN_TOA])) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        numBandPairs = len(list(context[Context.LIST_BAND_PAIR_INDICES]))
        meta.update(count=numBandPairs - 1)

        sr_prediction_list = list(context[Context.PRED_LIST])
        bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))

        ########################################
        # Read each layer and write it to stack
        ########################################
        with rasterio.open(output_name, 'w', **meta) as dst:
            for id in range(1, numBandPairs):
                bandPrediction = sr_prediction_list[id]
                dst.set_band_description(id, bandNamePairList[id - 1][1])
                bandPrediction1 = np.ma.masked_values(bandPrediction, -9999)
                dst.write_band(id, bandPrediction1)

        # Create Cloud-optimized Geotiff (COG)
        context[Context.FN_SRC] = str(output_name)
        context[Context.FN_DEST] = str(context[Context.FN_WARP])
        return self.createCOG(context)

    def createCOG(self, context):
        self._validateParms(context, [Context.FN_SRC, Context.CLEAN_FLAG,
                            Context.FN_DEST])

        # Use gdalwarp to create Cloud-optimized Geotiff (COG)
        cogname = context[Context.FN_SRC].replace("-precog.tif", ".tif")
        if eval(context[Context.CLEAN_FLAG]):
            if os.path.exists(context[Context.FN_DEST]):
                os.remove(context[Context.FN_DEST])
            if os.path.exists(cogname):
                os.remove(cogname)

        command = 'gdalwarp -of cog ' + context[Context.FN_SRC] + ' ' + cogname
        SystemCommand(command)
        if os.path.exists(context[Context.FN_SRC]):
            os.remove(context[Context.FN_SRC])
        return cogname

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

        if eval(context[Context.CLEAN_FLAG]):
            if os.path.exists(context[Context.FN_DEST]):
                os.remove(context[Context.FN_DEST])

        extent = self._getExtents(context[Context.TARGET_ATTR])
        ds = gdal.Warp(context[Context.FN_DEST], context[Context.FN_SRC],
                       dstSRS=context[Context.TARGET_SRS] , outputType=context[Context.TARGET_OUTPUT_TYPE] ,
                       xRes=context[Context.TARGET_XRES] , yRes=context[Context.TARGET_YRES], outputBounds=extent)
        ds = None

    def translate(self, context):
        self._validateParms(context, [Context.CLEAN_FLAG, Context.FN_DEST,
                                      Context.FN_SRC,
                                      Context.TARGET_XRES, Context.TARGET_YRES])

        if eval(context[Context.CLEAN_FLAG]):
            if os.path.exists(context[Context.FN_DEST]):
                os.remove(context[Context.FN_DEST])

        ds = gdal.Translate(context[Context.FN_DEST], context[Context.FN_SRC],
                       xRes=context[Context.TARGET_XRES] , yRes=context[Context.TARGET_YRES])
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
