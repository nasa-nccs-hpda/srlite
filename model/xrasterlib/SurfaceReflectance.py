import os
import numpy as np

# SR-Lite dependencies
from srlite.model.xrasterlib.Raster import Raster
from core.model.SystemCommand import SystemCommand
from osgeo import gdal, gdal_array, osr
from pygeotools.lib import warplib, geolib, iolib, malib, timelib
from srlite.model.regression.linear.SimpleLinearRegression import SimpleLinearRegression
import srlite.model.xrasterlib.RasterIndices as indices
from datetime import datetime  # tracking date
from rasterio.plot import show_hist
import matplotlib.pyplot as plt
import rasterio as rio  # geospatial library
from matplotlib import pyplot
import numpy.ma as ma
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
#import pandas as pd
#from tempfile import TemporaryDirectory

__author__ = "Glenn S. Tamkin, CISTO Branch"
__email__ = "glenn.s.tamkin@nasa.gov"
__status__ = "Development"

# -------------------------------------------------------------------------------
# class SR
# This class derives surface reflectance products. After performing a regression (e.g.,
# linear regression) across coincident model data (e.g., CCDC) and downscaled
# satellite imagery (e.g., WorldView), the derived coefficients are then applied to the
# originating high resolution imagery.  The full application is called "SR-Lite".
# -------------------------------------------------------------------------------

class SurfaceReflectance(Raster):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, model_image=None, low_res_image=None,
                 high_res_image=None, outdir='results'
                 ):
        super().__init__()

        # working directory to store result artifacts
        self.outdir = outdir

        # model path and filename (e.g. ccdc output)
        if model_image is not None and not os.path.isfile(model_image):
            raise RuntimeError('{} does not exist'.format(model_image))
        self.model_image = model_image

        # low resolution path and filename (e.g. 30m WV)
        if low_res_image is not None and not os.path.isfile(low_res_image):
            raise RuntimeError('{} does not exist'.format(low_res_image))
        self.low_res_image = low_res_image

        # high resolution path and filename (e.g. 30m WV)
        if high_res_image is not None and not os.path.isfile(high_res_image):
            raise RuntimeError('{} does not exist'.format(high_res_image))
        self.high_res_image = high_res_image

        # set default values
        self._targetBBox = None
        self._targetSRS = "ESRI:102001"
        self._targetNodata = -9999
        self._targetXres = 30
        self._targetYres = 30
        self._targetResampling = "average"

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------
    def extract_extents(self, image, args):
        """
        :param date: date of interest
        :param bounding_box: bounding box
        :return: validated date and bounding box
        """
        # TODO: Validate date and bounding box values
        if args.doi is None:
            raise RuntimeError('date not provided')
        else:  # if a date is given
            self.doi = args.doi

        if args.bbox is None:
            # get default extents from rw ccdc image
            modelDs = gdal.Open(image, gdal.GA_ReadOnly)
            extent = geolib.ds_geom_extent(modelDs)
            self.bbox = extent
            modelDs = None

        return self.doi, self.bbox;

    def get_ccdc_image(self, doi):
        """ Call GEE CCDC Service"""

        # TODO - add remote call to GEE
        modelDs = gdal.Open(self.model_image, gdal.GA_ReadOnly)
        extent = geolib.ds_geom_extent(modelDs)
        self.model_extent = extent
        self.model_proj = modelDs.GetProjection()
        self.model_geotrans = modelDs.GetGeoTransform()
        _, xres, _, _, _, yres = modelDs.GetGeoTransform()
        self.model_xres = xres
        self.model_yres = yres
        modelDs = None

        return self.model_image

    def get_evhr_image(self, doi):
        """ Call EVHR Service"""

        image = self.high_res_image
        if (self.low_res_image != None):
            image = self.low_res_image

        # TODO - add remote call to EVHR
        imageDs = gdal.Open(image, gdal.GA_ReadOnly)
        extent = geolib.ds_geom_extent(imageDs)
        self.image_extent = extent
        self.image_proj = imageDs.GetProjection()
        self.image_geotrans = imageDs.GetGeoTransform()
        _, xres, _, _, _, yres = imageDs.GetGeoTransform()
        self.image_xres = xres
        self.image_yres = yres
        imageDs = None

        return image

    def warp_image(self, input_image=None,
                   nodata_value=None, srs=None, bbox=None,
                   xres=None, yres=None, resampling=None, overwrite=False):
        """
        :param testsize: size of testing features
        :param seed: random state integer for reproducibility
        :return: 4 arrays with train and test data respectively
        """
        #gdalwarp -t_srs ESRI:102001 -te  -2927850 4063820 -2903670 4082450 -tr 30 -30 -r average

         # high resolution path and filename (e.g. 30m WV)
        if input_image is not None and not os.path.isfile(input_image):
            raise RuntimeError('{} does not exist'.format(input_image))

        #  Derive file names for potential intermediate files
        head, tail = os.path.split(input_image)
        filename = (tail.rsplit(".", 1)[0])
        filename_wo_ext = head + "/_" + filename + '-30m'
        warpFn = filename_wo_ext + '.tif'
        edited_image = warpFn

        if ((not os.path.isfile(warpFn)) or (overwrite == True)):
            #
            # Transform image (only apply parameters that are overridden)
            # if different SRS, user gdal_edit() to edit in place
            editParms = ""
            if (nodata_value != None):
                dataset = gdal.Open(input_image)
                imageNdv = dataset.GetRasterBand(1).GetNoDataValue()
                dataset = None
                nodata_flt_value = float(nodata_value)
                if (str(imageNdv) != str(float(nodata_flt_value))):
                    editParms = editParms +  ' -a_nodata ' + str(nodata_flt_value)
            if (srs != None):
                editParms = '' + editParms +  ' -t_srs ' + str(srs)
            if (bbox != None):
                editParms = '' + editParms +  ' -te ' + \
                            str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3]))
            if (xres != None):
                editParms = '' + editParms +  ' -tr ' + str(int(xres)) + ' ' + str(int(yres))
            if (resampling != None):
                editParms = '' + editParms +  ' -r ' + str(resampling)

            if bool(editParms) != False:
                # Use gdal_edit (via ILAB core SystemCommand) to convert GEE CCDC output to proper projection ESRI:102001 and set NoData value in place
                command = '/usr/bin/gdalwarp ' + editParms + ' ' + input_image + ' ' + warpFn
                SystemCommand(command)

            # Replace 'nan' values with nodata value (i.e., -9999) and store to disk
            edited_image = self.clean_image(warpFn, filename_wo_ext, nodata_value)

        return edited_image

    def edit_image(self, input_image=None,
                   nodata_value=None, srs=None,
                   xres=None, yres=None):
        """
        :param testsize: size of testing features
        :param seed: random state integer for reproducibility
        :return: 4 arrays with train and test data respectively
        """
         # high resolution path and filename (e.g. 30m WV)
        if input_image is not None and not os.path.isfile(input_image):
            raise RuntimeError('{} does not exist'.format(input_image))

        #  Derive file names for potential intermediate files
        head, tail = os.path.split(input_image)
        filename = (tail.rsplit(".", 1)[0])
        filename_wo_ext = head + "/_" + filename

        #
        # Transform image (only apply parameters that are overridden)
        # if different SRS, user gdal_edit() to edit in place
        editParms = ""
        if (nodata_value != None):
            dataset = gdal.Open(input_image)
            imageNdv = dataset.GetRasterBand(1).GetNoDataValue()
            dataset = None
            nodata_flt_value = float(nodata_value)
            if (str(imageNdv) != str(float(nodata_flt_value))):
                editParms = editParms +  ' -a_nodata ' + str(nodata_flt_value)
        if (srs != None):
            editParms = '' + editParms +  ' -a_srs ' + str(srs)
        if (xres != None):
            editParms = '' + editParms +  ' -tr ' + str(xres) + ' ' + str(yres)

        if bool(editParms) != False:
            # Use gdal_edit (via ILAB core SystemCommand) to convert GEE CCDC output to proper projection ESRI:102001 and set NoData value in place
            command = '/usr/bin/gdal_edit.py ' + editParms + ' ' + input_image
            SystemCommand(command)

        # Replace 'nan' values with nodata value (i.e., -9999) and store to disk
        edited_image = self.clean_image(input_image,filename_wo_ext, nodata_value)

        return edited_image

    # Replace 'nan' values with nodata value (i.e., -9999) and store to disk
    def clean_image(self, input_image, filename_wo_ext, nodata_value):

            edited_image = input_image

            # Use gdal to load original CCDC datafile as 3-dimensional array
            editArr = gdal_array.LoadFile(input_image)

            # Replace 'nan' values with nodata value (i.e., -9999) and store to disk
            array_sum = np.sum(editArr)
            array_has_nan = np.isnan(array_sum)

            if array_has_nan:
                editNoNanFn = filename_wo_ext + '-edit-nonan.tif'
                editArrNoNan = np.nan_to_num(editArr, copy=True, nan=nodata_value)

                # Save the file to disk as required for gdal_translate() commands below:
                clippedDs = self.save_rasters(
                    editArrNoNan,
                    input_image,
                    editNoNanFn,
                    self._targetNodata)
                edited_image = editNoNanFn
                clippedDs = None

            return edited_image

    def _display_plots(self, title, fn_list, warp_ma_masked_list):

        figsize = (10, 5)
        fig, axa = plt.subplots(nrows=1, ncols=len(fn_list), figsize=figsize, sharex=False, sharey=False)
        for i, ma in enumerate(warp_ma_masked_list):
            f_name = fn_list[i]
            divider = make_axes_locatable(axa[i])
            cax = divider.append_axes('right', size='2.5%', pad=0.05)
            im1 = axa[i].imshow(ma, cmap='RdYlGn', clim=malib.calcperc(ma, perc=(1, 95)))
            cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend='max')
            axa[i].set_title(title + os.path.split(f_name)[1], fontsize=10)
            cb.set_label('Reflectance (%)')

        plt.tight_layout()

        figsize = (10, 3)
        fig, axa = plt.subplots(nrows=1, ncols=len(warp_ma_masked_list), figsize=figsize, sharex=True, sharey=True)

        for i, ma in enumerate(warp_ma_masked_list):
            f_name = os.path.split(fn_list[i])[1]
            print(f" {ma.count()} valid pixels in WARPED MASKED ARRAY WITH COMMON MASK APPLIED version of {f_name}")

            h = axa[i].hist(ma.compressed(), bins=512, alpha=0.75)
            axa[i].set_title(title + f_name, fontsize=10)

        plt.tight_layout()

    def validate_intersection(self, warp_ds_list):

        ccdc_warp_ds, evhr_warp_ds = warp_ds_list

        # Get number of bands from CCDC image (assuems equivalent to
        ccdcNumBands = ccdc_warp_ds.RasterCount
        evhrNumBands = evhr_warp_ds.RasterCount
        if not ccdcNumBands == evhrNumBands:
#            raise RuntimeError('{} model band count does not match image after intersection'.format(evhrNumBands))
            print(f'WARNING: {ccdcNumBands} model band count does not match image {evhrNumBands} after intersection')
        ccdcExtent = geolib.ds_extent(ccdc_warp_ds)
        evhrExtent = geolib.ds_extent(evhr_warp_ds)
        if not ccdcExtent == evhrExtent:
            raise RuntimeError('{} model extent does not match image after intersection'.format(evhrExtent))

        if geolib.ds_IsEmpty(ccdc_warp_ds) or geolib.ds_IsEmpty(evhr_warp_ds):
            raise RuntimeError('{} model or image is empty after intersection'.format(evhrExtent))

        return ccdcNumBands

    def get_intersection(self, fn_list):

        ccdcNdv = iolib.get_ndv_fn(fn_list[0])
        evhrNdv = iolib.get_ndv_fn(fn_list[1])
        if (not ccdcNdv == self._targetNodata or (not evhrNdv == self._targetNodata)):
            raise RuntimeError('{} missing NoDataValue or not equal to: '.format(self._targetNodata))

        #dt_list = timelib.get_dt_list(fn_list)
        #s = malib.DEMStack(fn_list, res='min', extent='intersection')

        #ma_list = [iolib.fn_getma(fn) for fn in s]
       # self._display_plots("Before Intersection Results: ", fn_list, ma_list)

        ma_list = [iolib.fn_getma(fn) for fn in fn_list]
        self._display_plots("Before Intersection Results: ", fn_list, ma_list)

        # To intersect these images, use warplib - warp to CCDC resolution, extent, and srs
        warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='near')

        # Validate intersection
        ccdcNumBands = self.validate_intersection(warp_ds_list)

        geomIntersection = geolib.ds_geom_intersection(warp_ds_list)
        print(f'geom intersection of {fn_list[0]} and {fn_list[1]} = \n {geomIntersection}')

         # Get masked array from dataset
        warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]
        ccdc_warp_ma, evhr_warp_ma = warp_ma_list
        print(f'model shape {ccdc_warp_ma.shape} and image shape {evhr_warp_ma.shape}')
        print('\n',
              ccdc_warp_ma.shape,
              evhr_warp_ma.shape
              )

        self._display_plots("After Intersection Results: ", fn_list, warp_ma_list)

        # Get a common mask of valid data from the 2 inputs
        common_mask = malib.common_mask(warp_ma_list)
        common_warp_ma_masked_list = [np.ma.array(ccdc_warp_ma, mask=common_mask), np.ma.array(evhr_warp_ma, mask=common_mask)]
        self._display_plots("Common Mask Results: ", fn_list, common_warp_ma_masked_list)

        ccdc_warp_ds, evhr_warp_ds = warp_ds_list
        coefficients = [ccdcNumBands]
        index = 0
        while index < ccdcNumBands:
            index += 1
# ???            pygeotools.lib.iolib.gdal2np_dtype(b)
            ccdcBandMaArray = iolib.ds_getma(ccdc_warp_ds, index)
            evhrBandMaArray = iolib.ds_getma(evhr_warp_ds, index)
            warp_ma_masked_list = [np.ma.array(ccdcBandMaArray, mask=common_mask), np.ma.array(evhrBandMaArray, mask=common_mask)]

            self._display_plots("Single Band Mask Results" + str(index) + ": ", fn_list, warp_ma_masked_list)

#            slope, intercept, detrended_std = malib.ma_linreg(warp_ma_masked_list, dt_list)

#            slope, intercept, r_value, p_value, std_err = stats.linregress(warp_ma_masked_list[0], warp_ma_masked_list[1])

            # print("slope:", slope,
            #       "\nintercept:", intercept,
            #       "\nr squared:", r_value ** 2)

            lr = SimpleLinearRegression(warp_ma_masked_list[0], warp_ma_masked_list[1])
            coefficients.append(lr.run())

        print(f'Coefficients={coefficients}')

        return coefficients


        # Name the datasets
        modelDsAfterMemwarp, imageryDsAfterMemwarp = warp_ma_list

        #  Derive file names for intermediate files
        head, tail = os.path.split(fn_list[0])
        filename = (tail.rsplit(".", 1)[0])
        model_filename_wo_ext = head + "/_" + filename
        modelHighResIntersectionFn= model_filename_wo_ext + '-intersection.tif'

        # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
#        modelDsAfterMemwarpArr = gdal_array.DatasetReadAsArray(modelDsAfterMemwarp)
#        self.save_rasters(modelDsAfterMemwarpArr, fn_list[0], modelHighResIntersectionFn)
#        self._torasterBands(modelDsAfterMemwarpArr, fn_list[0], modelHighResIntersectionFn)

        #  Derive file names for intermediate files
        head, tail = os.path.split(fn_list[1])
        filename = (tail.rsplit(".", 1)[0])
        imagefilename_wo_ext = head + "/" + filename
        imageryHighResIntersectionFn= imagefilename_wo_ext + '-intersection.tif'

        # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
#        imageryDsAfterMemwarpArr = gdal_array.DatasetReadAsArray(imageryDsAfterMemwarp)
#        self.save_rasters(imageryDsAfterMemwarpArr, fn_list[1], imageryHighResIntersectionFn)

        # Save intermediate intersection file for convenience
        fn_list[0] = modelHighResIntersectionFn;
        fn_list[1] = imageryHighResIntersectionFn;

        return fn_list, warp_ma_list

    def _get_intersection(self, fn_list):

        modelDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
        imageryDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)

        #  Call memwarp to get the 'intersection' of the CCDC & EVHR datasets
        ds_list = warplib.memwarp_multi(
#            [modelDs, imageryDs], res='max',
            [modelDs, imageryDs], res=30,
            extent='intersection', t_srs='first', r='average', dst_ndv=self._targetNodata)

        # Name the datesets
        modelDsAfterMemwarp = ds_list[0]
        imageryDsAfterMemwarp = ds_list[1]

        #  Derive file names for intermediate files
        head, tail = os.path.split(fn_list[0])
        filename = (tail.rsplit(".", 1)[0])
        model_filename_wo_ext = head + "/_" + filename
        modelHighResIntersectionFn= model_filename_wo_ext + '-intersection.tif'

        # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
        modelDsAfterMemwarpArr = gdal_array.DatasetReadAsArray(modelDsAfterMemwarp)
        self.save_rasters(modelDsAfterMemwarpArr, fn_list[0], modelHighResIntersectionFn)

        #  Derive file names for intermediate files
        head, tail = os.path.split(fn_list[1])
        filename = (tail.rsplit(".", 1)[0])
        imagefilename_wo_ext = head + "/" + filename
        imageryHighResIntersectionFn= imagefilename_wo_ext + '-intersection.tif'

        # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
        imageryDsAfterMemwarpArr = gdal_array.DatasetReadAsArray(imageryDsAfterMemwarp)
        self.save_rasters(imageryDsAfterMemwarpArr, fn_list[1], imageryHighResIntersectionFn)

        # Save intermediate intersection file for convenience
        fn_list[0] = modelHighResIntersectionFn;
        fn_list[1] = imageryHighResIntersectionFn;

        modelDs = imageryDs = None
        return fn_list

        # model2Ds = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
        # imagery2Ds = gdal.Open(fn_list[1], gdal.GA_ReadOnly)
        #
        # #  SWITCH order - Call memwarp to get the 'intersection' of the CCDC & EVHR datasets
        # ds_list2 = warplib.memwarp_multi(
        #     [imagery2Ds, model2Ds], res=30,
        #     extent='intersection', t_srs='first', r='average', dst_ndv=self._targetNodata)
        #
        # modelHighResIntersectionFn2 = model_filename_wo_ext + '-intersection2.tif'
        # imageryHighResIntersectionFn2 = imagefilename_wo_ext + '-intersection2.tif'
        #
        # # Name the datesets - NOTE order reversed from 1st call
        # modelDsAfterMemwarp2 = ds_list2[0]
        # imageryDsAfterMemwarp2 = ds_list2[1]
        #
        # # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
        # modelDsAfterMemwarpArr2 = gdal_array.DatasetReadAsArray(modelDsAfterMemwarp2)
        # self.save_rasters(modelDsAfterMemwarpArr2, modelHighResIntersectionFn, modelHighResIntersectionFn2)
        #
        # imageryDsAfterMemwarpArr2 = gdal_array.DatasetReadAsArray(imageryDsAfterMemwarp2)
        # self.save_rasters(imageryDsAfterMemwarpArr2, imageryHighResIntersectionFn, imageryHighResIntersectionFn2)
        #
        # modelDs = imageryDs = modelDs2 = imageryDs2 = None
        #
        # # Save intermediate intersection file for convenience
        # fn_list[0] = modelHighResIntersectionFn2;
        # fn_list[1] = imageryHighResIntersectionFn2;

        # modelDs = imageryDs = modelDs2 = imageryDs2 = None
        # return fn_list

    def build_model(self, warp_ma_list):

            ccdcDs = warp_ma_list[0]
            evhrDs = warp_ma_list[1]

            numBands = 4
#            numBands = ccdcDs.RasterCount
            coefficients = [numBands]

            # Assumes 1:1 mapping of band-ordering across model and imagery
            for index in range(1, numBands + 1):
                # read in bands from image
                ccdcBand = ccdcDs.GetRasterBand(index)
                ccdcNodata = ccdcBand.GetNoDataValue()
                ccdcArray = ccdcBand.ReadAsArray()
                ccdcMaArray = np.ma.masked_equal(ccdcArray, ccdcNodata)

                evhrBand = evhrDs.GetRasterBand(index)
                evhrNodata = evhrBand.GetNoDataValue()
                evhrArray = evhrBand.ReadAsArray()
                evhrMaArray = np.ma.masked_equal(evhrArray, evhrNodata)

                # get coefficients
                #            lr = SimpleLinearRegression(evhrMaArray, ccdcMaArray)
                lr = SimpleLinearRegression(ccdcMaArray, evhrMaArray)
                coefficients.append(lr.run())

            # ax.legend()
            # ax.grid(True)
            # plt.show()

            ccdcDs = evhrDs = None

            return coefficients

    def _build_model(self, fn_list):

        # show histogram
#        self.show_combo(fn_list[0])

#        self.show_hist(fn_list[0])
#        self.show_hist(fn_list[1])

        ccdcDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
        evhrDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)

        numBands = ccdcDs.RasterCount
        coefficients = [numBands]

#        fig, ax = plt.subplots()

        # Assumes 1:1 mapping of band-ordering across model and imagery
        for index in range(1, numBands + 1):
            # read in bands from image
            ccdcBand = ccdcDs.GetRasterBand(index)
            ccdcNodata = ccdcBand.GetNoDataValue()
            ccdcArray = ccdcBand.ReadAsArray()

            # temporarily force nodatavalue out of array
#            ccdcArray = ccdcArray[ccdcArray > 0]

            ccdcMaArray = np.ma.masked_equal(ccdcArray, ccdcNodata)

            evhrBand = evhrDs.GetRasterBand(index)
            evhrNodata = evhrBand.GetNoDataValue()
            evhrArray = evhrBand.ReadAsArray()

            # temporarily force nodatavalue out of array
#            evhrArray = evhrArray[evhrArray > 0]

            evhrMaArray = np.ma.masked_equal(evhrArray, evhrNodata)

            # get intersection of masded arrays
            ccdcMask = ccdcMaArray.mask
            evhrMask = evhrMaArray.mask
            ccdcMaArray[ccdcMask] = ccdcMaArray[evhrMask]

            #            self.masked_plot(ccdcArray, ccdcMaArray, evhrArray, evhrMaArray)
    #            ax.scatter(ccdcArray, evhrArray,
#                       alpha=0.3, edgecolors='none')
#            self.scatter_hist(ccdcMaArray, evhrMaArray)
            import numpy.ma as ma
            print(f"Non-masked & Masked value ccdc = total... {ma.count(ccdcMaArray)} {ma.count_masked(ccdcMaArray)} " \
                  f"{ma.count(ccdcMaArray) + ma.count_masked(ccdcMaArray)}")
            print(f"Non-masked & Masked value evhr = total...{ma.count(evhrMaArray)} {ma.count_masked(evhrMaArray)} " \
                      f"{ma.count(evhrMaArray) + ma.count_masked(evhrMaArray)}")

            # get coefficients
#            lr = SimpleLinearRegression(evhrMaArray, ccdcMaArray)
            lr = SimpleLinearRegression(ccdcMaArray, evhrMaArray)
            coefficients.append(lr.run())

        # ax.legend()
        # ax.grid(True)
        # plt.show()

        ccdcDs = evhrDs = None

        return coefficients

    def build_stats_model(self, fn_list):

        ccdcDs = gdal.Open(fn_list[0], gdal.GA_ReadOnly)
        evhrDs = gdal.Open(fn_list[1], gdal.GA_ReadOnly)

        numBands = ccdcDs.RasterCount
        coefficients = [numBands]

        # Assumes 1:1 mapping of band-ordering across model and imagery
        for index in range(1, numBands + 1):
            # read in bands from image
            ccdcBand = ccdcDs.GetRasterBand(index)
            ccdcNodata = ccdcBand.GetNoDataValue()
            ccdcArray = ccdcBand.ReadAsArray()
            ccdcMaArray = np.ma.masked_equal(ccdcArray, ccdcNodata)
            ccdcMaArrayFlat = ccdcMaArray.flatten()

            evhrBand = evhrDs.GetRasterBand(index)
            evhrNodata = evhrBand.GetNoDataValue()
            evhrArray = evhrBand.ReadAsArray()
            evhrMaArray = np.ma.masked_equal(evhrArray, evhrNodata)
            evhrMaArrayFlat = evhrMaArray.flatten()

            # ccdcMaArrayDf = pd.DataFrame(ccdcMaArrayFlat)
            # evhrMaArrayDf = pd.DataFrame(evhrMaArrayFlat)
            # slope, intercept, r_value, p_value, std_err = stats.linregress(ccdcMaArrayDf, evhrMaArrayDf)
            #
            # print("slope:", slope,
            #       "\nintercept:", intercept,
            #       "\nr squared:", r_value ** 2)
            print(
                f"Flat Non-masked & Masked value ccdc = total... {ma.count(ccdcMaArrayFlat)} {ma.count_masked(ccdcMaArrayFlat)} " \
                    f"{ma.count(ccdcMaArrayFlat) + ma.count_masked(ccdcMaArrayFlat)}")
            print(
                f"Flat Non-masked & Masked value evhr = total...{ma.count(evhrMaArrayFlat)} {ma.count_masked(evhrMaArrayFlat)} " \
                    f"{ma.count(evhrMaArrayFlat) + ma.count_masked(evhrMaArrayFlat)}")

            # get coefficients
            #            lr = SimpleLinearRegression(evhrMaArray, ccdcMaArray)
#            lr = SimpleLinearRegression(ccdcMaArray, evhrMaArray)
            lr = SimpleLinearRegression(ccdcMaArrayFlat, evhrMaArrayFlat)
            coefficients.append(lr.run())

        # ax.legend()
        # ax.grid(True)
        # plt.show()

        ccdcDs = evhrDs = None

        return coefficients

    def apply_model(self, image, coefficients, args):

        rast = image
        print(f"Apply coefficients...{rast}")
        self.readraster(rast, args.bands_data)  # read raster

        # add transformed WV Bands
        # B1(Blue): 450 - 510
        # B2(Green): 510 - 580
        # B3(Red): 655 - 690
        # B4(NIR): 780 - 920

        srcNumBands = self.data.shape[0]
        self.requestedNumBands = len(args.bands_data)

        print(f"Original Data size {srcNumBands} and Requested Data Size {self.requestedNumBands} contents {args.bands_data}")

        # high resolution path and filename (e.g. 30m WV)
        if srcNumBands != self.requestedNumBands:
#            raise RuntimeError(f'Original Data size {srcNumBands} and Requested Data Size {self.requestedNumBands} must match')
            print(f'Original Data size {srcNumBands} and Requested Data Size {self.requestedNumBands} must match')

        # See indices.py (b1-b4) for LR coefficients implementation: y = ee.Number(x).multiply(slope).add(yInt);
        self.xformBands(self.data, coefficients, args.bands_model )

        # out mask name, save raster
        now = datetime.now()  # current date and time
        nowStr = now.strftime("%b:%d:%H:%M:%S")
        # print("time:", nowStr)
        #  Derive file names for intermediate files
        head, tail = os.path.split(image)
        filename = (tail.rsplit(".", 1)[0])
        output_name = "{}/srlite-{}-{}".format(
            self.outdir, filename,
            nowStr, rast.split('/')[-1]
        ) + ".tif"

        #            print(f"Max of raster {raster_obj.data.max}")
        self.torasterBands(args, rast, self.data, output_name)

        return output_name

    def loadMetadata(self, image):
        # open the raster and its spatial reference
        self.src = gdal.Open(image)

        if self.src is None:
            raise Exception('Could not load GDAL file "%s"' % image)
        spatial_reference_raster = osr.SpatialReference(self.src.GetProjection())

        # get the WGS84 spatial reference
        spatial_reference = osr.SpatialReference()
        spatial_reference.ImportFromEPSG(4326)  # WGS84

        # coordinate transformation
        self.coordinate_transform = osr.CoordinateTransformation(spatial_reference, spatial_reference_raster)
        gt = self.geo_transform = self.src.GetGeoTransform()
        dev = (gt[1] * gt[5] - gt[2] * gt[4])
        self.geo_transform_inv = (gt[0], gt[5] / dev, -gt[2] / dev,
                                  gt[3], -gt[4] / dev, gt[1] / dev)

    def reproject_geotransform(self, in_gt, old_proj_wkt, new_proj_wkt):
        """
        Reprojects a geotransform from the old projection to a new projection. See
        [https://gdal.org/user/raster_data_model.html]

        Parameters
        ----------
        in_gt
            A six-element numpy array, usually an output from gdal_image.GetGeoTransform()
        old_proj_wkt
            The projection of the old geotransform in well-known text.
        new_proj_wkt
            The projection of the new geotrasform in well-known text.
        Returns
        -------
        out_gt
            The geotransform in the new projection

        """
        modelDs = gdal.Open(self.model_image, gdal.GA_ReadOnly)
        extent = geolib.ds_geom_extent(modelDs)
        self.model_extent = extent
        self.model_proj = modelDs.GetProjection()
        self.model_geotrans = modelDs.GetGeoTransform()
        modelDs = None

        imageDs = gdal.Open(self.high_res_image, gdal.GA_ReadOnly)
        extent = geolib.ds_geom_extent(imageDs)
        self.image_extent = extent
        self.image_proj = imageDs.GetProjection()
        self.image_geotrans = imageDs.GetGeoTransform()
        imageDs = None

        in_gt = self.model_geotrans
        old_proj = osr.SpatialReference()
        new_proj = osr.SpatialReference()
        old_proj.ImportFromWkt(self.model_proj)
        new_proj.ImportFromWkt(self.image_proj)
        transform = osr.CoordinateTransformation(old_proj, new_proj)
        (ulx, uly, _) = transform.TransformPoint(in_gt[0], in_gt[3])
        out_gt = (ulx, in_gt[1], in_gt[2], uly, in_gt[4], in_gt[5])
        return out_gt

    def scatter_hist(self, x, y):
#        def scatter_hist(x, y, ax, ax_histx, ax_histy):

        # Create a Figure, which doesn't have to be square.
        fig = plt.figure(constrained_layout=True)
        # Create the main axes, leaving 25% of the figure space at the top and on the
        # right to position marginals.
        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        # The main axes' aspect can be fixed.
        ax.set(aspect=1)
        # Create marginal axes, which have 25% of the size of the main axes.  Note that
        # the inset axes are positioned *outside* (on the right and the top) of the
        # main axes, by specifying axes coordinates greater than 1.  Axes coordinates
        # less than 0 would likewise specify positions on the left and the bottom of
        # the main axes.
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
        # Draw the scatter plot and marginals.

        #scatter_hist(x, y, ax, ax_histx, ax_histy)

        plt.show()
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')

    def masked_plot(self, x_values, x_values_masked, y_values, y_values_masked):
        # give a threshold
        threshold = 1

        # plot all data
        plt.subplots_adjust(hspace=0.5)
        plt.subplot(311)
        plt.plot(x_values, y_values, 'ko')
        plt.title('All values')
        plt.subplot(312)
        plt.plot(x_values, y_values_masked, 'ko')
        plt.title('Plot without masked values')
        ax = plt.subplot(313)
        ax.plot(x_values, y_values_masked, 'ko')
        # for otherwise the range of x_values gets truncated:
        ax.set_xlim(x_values[0], x_values[-1])
        plt.title('Plot without masked values -\nwith full range x-axis')
        plt.show()

    def show_hist(self, rast):
        #
        with rio.open(rast) as src:
            hist_data = src.read([4, 3, 2])

            fig, axhist = plt.subplots(1, 1)
            show_hist(hist_data, bins=200, histtype='stepfilled',
                      lw=0.0, stacked=False, alpha=0.8, ax=axhist)
            axhist.set_xlabel('X')
            axhist.set_ylabel('Y')
            axhist.set_title('Reflectance Histogram')
            # plt.xlim(xmin=0, xmax=1)
            # plt.xlabel('Reflectance')
            # show_hist(his_data, bins=200, lw=0.0, stacked=False, alpha=0.8, histtype='stepfilled', title="Histogram")

    def show_combo(self, rast):
        with rio.open(rast) as src:
            pyplot.imshow(src.read(4), cmap='pink')
            pyplot.imshow(src.read(3), cmap='rainbow')
            pyplot.imshow(src.read(2), cmap='turbo')
#            pyplot.imshow(src.read([4, 3, 2]), cmap='pink')
#            fig, (axrgb, axhist) = pyplot.subplots(1, 2, figsize=(14, 7))
#            pyplot.show(src, ax=axrgb)
            self.show_hist(rast)
            #self.show_hist(rast, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3, ax=axhist)
#            pyplot.show()


 # -------------------------------------------------------------------------------
# class RF Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Running Unit Tests
    print("Unit tests below")
