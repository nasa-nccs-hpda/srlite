#!/usr/bin/env python
# coding: utf-8
import os
import sys
from datetime import datetime
import osgeo
from osgeo import gdal
from pygeotools.lib import iolib, warplib
import rasterio
import numpy as np

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
                 print(osgeo.gdal.VersionInfo())
        except BaseException as err:
            print('ERROR - check gdal version: ')
            print(err)
            sys.exit(1)
        return

    def getBandIndices(self, fn_list, bandNamePair):
        """
        Validate band name pairs and return corresponding gdal indices
        """
        ccdcDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
        ccdcBands = ccdcDs.RasterCount
        evhrDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)
        evhrBands = evhrDs.RasterCount

        self._plot_lib.trace('bandNamePair: ' + str(bandNamePair))
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

    def getProjection(self, r_fn, title):
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

        self._plot_lib.plot_combo(r_fn, figsize=(14, 7), title=title)

    def validateBands(self, bandNamePairList, fn_list):
        ########################################
        # Validate Band Pairs and Retrieve Corresponding Array Indices
        ########################################
        self._plot_lib.trace('bandNamePairList=' + str(bandNamePairList))
        #    fn_full_list = [r_fn_ccdc, r_fn_evhr]
        bandPairIndicesList = self.getBandIndices(fn_list, bandNamePairList)
        self._plot_lib.trace('bandIndices=' + str(bandPairIndicesList))
        return bandPairIndicesList

    def getIntersection(self, fn_list):
        # ########################################
        # # Align the CCDC and EVHR images, then take the intersection of the grid points
        # ########################################
        warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='average')
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        return warp_ds_list, warp_ma_list

    def createImage(self, r_fn_evhr, numBandPairs, sr_prediction_list, name,
                    bandNamePairList, outdir):
        ########################################
        # Create .tif image from band-based prediction layers
        ########################################
        self._plot_lib.trace(f"\nApplot_liby coefficients to High Res File...{r_fn_evhr}")

        now = datetime.now()  # current date and time
        nowStr = now.strftime("%m%d%H%M")

        #  Derive file names for intermediate files
        head, tail = os.path.split(r_fn_evhr)
        filename = (tail.rsplit(".", 1)[0])
        output_name = "{}/{}".format(
            outdir, name
        ) + "_sr_02m-precog.tif"
        self._plot_lib.trace(f"\nCreating .tif image from band-based prediction layers...{output_name}")

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
                self._plot_lib.trace(f'BandPrediction.mean({mean})')
                dst.set_band_description(id, bandNamePairList[id - 1][1])
                bandPrediction1 = np.ma.masked_values(bandPrediction, -9999)
                dst.write_band(id, bandPrediction1)

        return output_name

    def getProjSrs(self, in_raster):
        from osgeo import gdal, osr
        ds = gdal.Open(in_raster)
        prj = ds.GetProjection()
        print(prj)

        srs = osr.SpatialReference(wkt=prj)
        print('srs=', srs)
        if srs.IsProjected:
            print(srs.GetAttrValue('projcs'))
        print(srs.GetAttrValue('geogcs'))
        return prj, srs

    def getExtents(self, in_raster):
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

    def getMetadata(self, band_num, input_file):
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

    def warp(self, in_raster, outraster, dstSRS, outputType, xRes, yRes, extent):
        from osgeo import gdal
        ds = gdal.Warp(in_raster, outraster,
                       dstSRS=dstSRS, outputType=outputType, xRes=yRes, yRes=yRes, outputBounds=extent)
        ds = None
        return outraster

    def downscale(self, targetAttributesFile, inFile, outFile, xRes=30.0, yRes=30.0):
        import os.path
        if not os.path.exists(str(outFile)):
            outputType = self.getMetadata(1, str(targetAttributesFile))
            new_projection, new_srs = self.getProjSrs(targetAttributesFile)
            extent = self.getExtents(targetAttributesFile)
            print(extent)
            outFile = self.warp(outFile, inFile, dstSRS=new_srs, outputType=outputType, xRes=xRes, yRes=yRes, extent=extent)
            ds = None
        return outFile

    def applyThreshold(self, min, max, bandMaArray):
        ########################################
        # Mask threshold values (e.g., (median - threshold) < range < (median + threshold)
        #  prior to generating common mask to reduce outliers ++++++[as per MC - 02/07/2022]
        ########################################
        self._plot_lib.trace('======== Appying threshold algorithm to first EVHR Band (Assume Blue) ========================')
        bandMaThresholdMaxArray = np.ma.masked_where(bandMaArray > max, bandMaArray)
        bandMaThresholdRangeArray = np.ma.masked_where(bandMaThresholdMaxArray < min, bandMaThresholdMaxArray)
        self._plot_lib.trace(' threshold range median =' + str(np.ma.median(bandMaThresholdRangeArray)))
        return bandMaThresholdRangeArray
