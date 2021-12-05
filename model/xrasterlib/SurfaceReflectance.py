import os
import numpy as np

# SR-Lite dependencies
from model.xrasterlib.Raster import Raster
from core.model.SystemCommand import SystemCommand
from osgeo import gdal, gdal_array
from pygeotools.lib import warplib
from model.regression.linear.SimpleLinearRegression import SimpleLinearRegression
import model.xrasterlib.RasterIndices as indices
from datetime import datetime  # tracking date

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
        self._targetBBox = [-2927850, 4063820, -2903670, 4082450]
        self._targetSRS = "ESRI:102001"
        self._targetNodata = -9999
        self._targetXres = 30
        self._targetYres = 30

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------
    def extract_extents(self, args):
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
            raise RuntimeError('bounding box not provided')
        else:  # if a bounding_box is given
            self.bbox = args.bbox

        return args.doi, args.bbox;

    def edit_image(self, input_image=None,
                   nodata_value=None, srs=None, bbox=None,
                   xres=None, yres=None):
        """
        :param testsize: size of testing features
        :param seed: random state integer for reproducibility
        :return: 4 arrays with train and test data respectively
        """
         # high resolution path and filename (e.g. 30m WV)
        if input_image is not None and not os.path.isfile(input_image):
            raise RuntimeError('{} does not exist'.format(input_image))
        edited_image = input_image
        #
        # Transform image (only apply parameters that are overridden)
        editParms = ""
        if (nodata_value != None):
            dataset = gdal.Open(input_image)
            imageNdv = dataset.GetRasterBand(1).GetNoDataValue()
            if (str(imageNdv) != str(nodata_value)):
                editParms = editParms +  ' -a_nodata ' + str(nodata_value)
        if (srs != None):
            editParms = '' + editParms +  ' -a_srs ' + str(srs)

        if bool(editParms) != False:
            # Use gdal_edit (via ILAB core SystemCommand) to convert GEE CCDC output to proper projection ESRI:102001 and set NoData value in place
            command = 'gdal_edit.py ' + editParms + ' ' + input_image
            SystemCommand(command)

            # Use gdal to load original CCDC datafile as 3-dimensional array
            editArr = gdal_array.LoadFile(input_image)

            # Replace 'nan' values with nodata value (i.e., -9999) and store to disk
            editArrNoNan = np.nan_to_num(editArr, copy=True, nan=nodata_value)

            #  Derive file names for intermediate files
            head, tail = os.path.split(input_image)
            filename = (tail.rsplit(".", 1)[0])
            filename_wo_ext = head + "/_" + filename
            editNoNanFn = filename_wo_ext + '-edit-nonan.tif'
            editNoNanClipFn = filename_wo_ext + '-bbox.tif'
            editNoNanClipResFn = filename_wo_ext + '-bbox-res.tif'

            # Save the file to disk as required for gdal_translate() commands below:
            clippedDs = self.save_rasters(
                    editArrNoNan,
                    input_image,
                    editNoNanFn,
                    self._targetNodata)

            #  Apply bounding box and scale to CCDC data - API barked when I tried both in same gdal_translate() call
            extents = ' '.join((str(e) for e in bbox))
            command = \
                'gdal_translate -a_srs ' + str(srs) + ' -a_ullr ' + str(extents) + ' -a_nodata ' \
                + str(nodata_value) + ' ' + editNoNanFn + ' ' + editNoNanClipFn
            SystemCommand(command)

            #  Apply bounding box and scale to CCDC data - API barked when I tried both in same gdal_translate() call
            command = \
                'gdal_translate -tr ' + str(xres) + ' ' +  str(xres) +  ' ' \
                + editNoNanClipFn + ' ' + editNoNanClipResFn
            SystemCommand(command)

            # Return edited image - clipped, reprojected, and added nodata value
            edited_image = editNoNanClipResFn

        return edited_image

    def get_intersection(self, fn_list):

        modelDs = gdal.Open(fn_list[0], gdal.GA_Update)
        imageryDs = gdal.Open(fn_list[1], gdal.GA_Update)

        #  Call memwarp to get the 'intersection' of the CCDC & EVHR datasets
        ds_list = warplib.memwarp_multi(
#            [modelDs, imageryDs], res='max',
            [modelDs, imageryDs], res=30,
            extent='intersection', t_srs='first', r='average', dst_ndv=self._targetNodata)

        # Name the datesets
        modelDsAfterMemwarp = ds_list[0]
        imageryDsAfterMemwarp = ds_list[1]

        #  Derive file names for intermediate files
        head, tail = os.path.split(fn_list[1])
        filename = (tail.rsplit(".", 1)[0])
        filename_wo_ext = head + "/_" + filename
        imageryHighResIntersectionFn= filename_wo_ext + '-intersection.tif'

        # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
        imageryDsAfterMemwarpArr = gdal_array.DatasetReadAsArray(imageryDsAfterMemwarp)
        self.save_rasters(imageryDsAfterMemwarpArr, self.high_res_image, imageryHighResIntersectionFn)

        # Save intermediate intersection file for convenience
        self._intersection = imageryHighResIntersectionFn;

        return ds_list

    def build_model(self, ds_list):

        numBands = ds_list[0].RasterCount
        coefficients = [numBands]

        # Assumes 1:1 mapping of band-ordering across model and imagery
        for index in range(1, numBands + 1):
            # read in bands from image
            ccdcBand = ds_list[0].GetRasterBand(index)
            ccdcNodata = ccdcBand.GetNoDataValue()
            ccdcArray = ccdcBand.ReadAsArray()
            ccdcMaArray = np.ma.masked_equal(ccdcArray, ccdcNodata)

            evhrBand = ds_list[1].GetRasterBand(index)
            evhrNodata = evhrBand.GetNoDataValue()
            evhrArray = evhrBand.ReadAsArray()
            evhrMaArray = np.ma.masked_equal(evhrArray, evhrNodata)

            # get coefficients
            lr = SimpleLinearRegression(ccdcMaArray, evhrMaArray)
            coefficients.append(lr.run())

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

        # See indices.py (b1-b4) for LR coefficients implementation: y = ee.Number(x).multiply(slope).add(yInt);
        print(f"Size of raster {self.data.shape[0]} before indices")
        self.addindices(
            #            [indices.blue, indices.green, indices.red, indices.nir],
            [indices.b1, indices.b2, indices.b3, indices.b4]
        )
        # out mask name, save raster
        now = datetime.now()  # current date and time
        nowStr = now.strftime("%b:%d:%H:%M:%S")
        # print("time:", nowStr)
        output_name = "{}/srlite-{}-{}-{}{}".format(
            self.outdir, args.model, args.regression,
            nowStr, rast.split('/')[-1]
        )

        #            print(f"Max of raster {raster_obj.data.max}")
        self.torasterBands(rast, self.data, output_name)

        return output_name
    
    def get_ccdc_image(self, doi, bbox):
        """ Call GEE CCDC Service"""

        # TODO - add remote call to GEE
        return self.model_image

    def get_evhr_image(self, doi, bbox):
        """ Call EVHR Service"""

        # TODO - add remote call to EVHR
        return self.high_res_image

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__deprecated(self, traincsvfile=None, modelfile=None, outdir='results',
                 ntrees=20, maxfeat='log2'
                 ):
        super().__init__()

        # working directory to store result artifacts
        self.outdir = outdir

        # training csv filename
        if traincsvfile is not None and not os.path.isfile(traincsvfile):
            raise RuntimeError('{} does not exist'.format(traincsvfile))
        self.traincsvfile = traincsvfile

        # training parameters
        self.ntrees = ntrees
        self.maxfeat = maxfeat

        # training and test data variables, initialize them as empty
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # trained model filename
        # TODO: update this with args.command option (train vs. classify)
        if traincsvfile is None and modelfile is not None \
                and not os.path.isfile(modelfile):
            raise RuntimeError('{} does not exist'.format(modelfile))

        elif modelfile is None and self.traincsvfile is not None:
            self.modelfile = 'model_{}_{}.pkl'.format(
                self.ntrees, self.maxfeat
            )
        else:  # if a model name is given
            self.modelfile = modelfile

        # store loaded model
        self.model = None
        self.model_nfeat = None

        # store prediction if required
        self.prediction = None

# -------------------------------------------------------------------------------
# class RF Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Running Unit Tests
    print("Unit tests below")
