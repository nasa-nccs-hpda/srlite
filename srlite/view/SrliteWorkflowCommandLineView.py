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
from datetime import datetime  # tracking date
import time  # tracking time
#from time import time  # tracking time
import argparse  # system libraries
import numpy as np
import rasterio
from osgeo import gdal
from srlite.model.PlotLib import PlotLib
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

def create_logfile(args, logdir='results'):
    """
    :param args: argparser object
    :param logdir: log directory to store log file
    :return: logfile instance, stdour and stderr being logged to file
    """
    logfile = os.path.join(logdir, '{}_log_{}_model_{}_doi_{}.txt'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), args.model, args.regression, args.doi))
    print('See ', logfile)
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # stdout buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect to the log file
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


def getparser():
    """
    :return: argparser object with CLI commands.
    """
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "-d", "--date", nargs=1, type=str, required=True, dest='doi',
        default="2011-08-18",
        help = "Specify date to perform regression (YYYY-MM-DD)."
    )
    parser.add_argument(
        "-bb", "--bounding-box", nargs=4, type=int, required=False, dest='bbox',
        default=None,
        help = "Specify bounding box to perform regression."
    )
    parser.add_argument(
        "-i-m", "--input-model-image", type=str, required=False, dest='model_image',
        default=None,
        help="Specify model (e.g., CCDC) input image path and filename."
    )
    parser.add_argument(
        '-b-m', '--bands-model', nargs='*', dest='bands_model',
        default=['blue', 'green', 'redd', 'nir'], required=False, type=str,
        help = 'Specify input model (e.g., CCDC) bands.'
    )
    parser.add_argument(
        "-i-lr", "--input-low-res-image", type=str, required=False, dest='low_res_image',
        default=None,
        help="Specify low-resolution input image path and filename."
    )
    parser.add_argument(
        "-i-hr", "--input-high-res-image", type=str, required=True, dest='high_res_image',
        default=None,
        help="Specify high-resolution input image path and filename."
    )
    parser.add_argument(
        '-b-d', '--bands-data', nargs='*', dest='bands_data', required=True, type=str,
        default=['b1', 'b2ff', 'b3', 'b4'],
        help='Specify input data (e.g., HR WV) bands.'
    )
    parser.add_argument(
        "-m", "--model", type=str, required=False, dest='model',
        default='ccdc', help="Specify model to run."
    )
    parser.add_argument(
        "-r", "--regression", type=str, required=False, dest='regression',
        default='linear-regression', help="Specify regression to run."
    )
    parser.add_argument(
        "-o", "--out-directory", type=str, required=True, dest='outdir',
        default="", help="Specify output directory."
    )
    parser.add_argument(
        "-fo", "--force-overwrite", required=False, dest='force_overwrite',
        default=False, action='store_true', help="Force overwrite."
    )
    parser.add_argument(
        "-l", "--log", required=False, dest='logbool',
        action='store_true', help="Set logging."
    )
    return parser.parse_args()

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
                bandNamePairList,outdir, pl):
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


def processBands(warp_ds_list, bandPairIndicesList, fn_list, bandNamePairList, override, pl):
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

    firstBand = True
    # Apply range of -100 to 200 for Blue Band pixel mask and apply to each band (as per MC - 3/1/22 - for big batch)
    evhrBandMaArrayThresholdMin = -100
    evhrBandMaArrayThresholdMax = 2000
    pl.trace(' evhrBandMaArrayThresholdMin = ' + str(evhrBandMaArrayThresholdMin))
    pl.trace(' evhrBandMaArrayThresholdMax = ' + str(evhrBandMaArrayThresholdMax))

    for bandPairIndex in range(0, numBandPairs - 1):

        pl.trace('=>')
        pl.trace('====================================================================================')
        pl.trace('============== Start Processing Band #' + str(bandPairIndex + 1) + ' ===============')
        pl.trace('====================================================================================')

        # Retrieve band pair
        bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

        # Get 30m CCDC Masked Arrays
        ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])

        # Get 2m EVHR Masked Arrays
        evhrBandMaArrayRaw = iolib.fn_getma(fn_list[1], bandPairIndices[1])

        #  Create single mask for all bands based on Blue-band threshold values
        #  Assumes Blue-band is first indice pair, so collect mask on 1st iteration only.
        if (firstBand == True):
            ########################################
            # Mask threshold values (e.g., (median - threshold) < range < (median + threshold)
            #  prior to generating common mask to reduce outliers ++++++[as per MC - 02/07/2022]
            ########################################
            evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

            # Logic below applies dynamic Min & Max threshold range based on Median
            # evhrBandMaArrayMedian = np.ma.median(evhrBandMaArray)
            # pl.trace(' evhrBandMaArrayMedian median =' + str(np.ma.median(evhrBandMaArrayMedian)))
            # threshold = 500
            # evhrBandMaArrayThresholdMin = evhrBandMaArrayMedian - threshold
            # evhrBandMaArrayThresholdMax = evhrBandMaArrayMedian + threshold

            # Apply upper bound to EVHR values in raw 2m array
            evhrBandMaThresholdMaxArray = \
                np.ma.masked_where(evhrBandMaArray > evhrBandMaArrayThresholdMax, evhrBandMaArray)

            # Apply lower bound to EVHR values to modified 2m array above
            evhrBandMaThresholdRangeArray = \
                np.ma.masked_where(evhrBandMaThresholdMaxArray < evhrBandMaArrayThresholdMin,
                                   evhrBandMaThresholdMaxArray)
            pl.trace(' evhrBandMaThresholdRangeArray median =' + str(np.ma.median(evhrBandMaThresholdRangeArray)))

            evhrBandMaArray = evhrBandMaThresholdRangeArray
            firstBand = False

        # Generate common mask
        warp_ma_band_list = [ccdcBandMaArray, evhrBandMaArray]
        warp_valid_ma_band_list = warp_ma_band_list
        common_mask_band = malib.common_mask(warp_valid_ma_band_list)

        warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band),
                                    np.ma.array(evhrBandMaArray, mask=common_mask_band)]

        # Check the mins of each ma - they should be greater than <evhrBandMaArrayThresholdMin>
        for j, ma in enumerate(warp_ma_masked_band_list):
            j = j + 1
            #            if (ma.min() < 0):
            if (ma.min() < int(evhrBandMaArrayThresholdMin)):
                pl.trace("Warning: Masked array values should be larger than " + str(evhrBandMaArrayThresholdMin))
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
                           title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR DELTA ",
                           override=override)
        pl.plot_maps([evhr_pre_post_ma_list[1],
                      evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]],
                     [compare_name_list[1],
                      str(bandNamePairList[bandPairIndex]) + ' Difference: TOA-SR-Lite'], (10, 50),
                     cmap_list=['RdYlGn', 'RdBu'], override=override)
        print(f"\nFinished with {str(bandNamePairList[bandPairIndex])} Band")

    return sr_prediction_list


# ## Set up inputs

# In[8]:


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():

     # --------------------------------------------------------------------------------
    # 0. Prepare for run - set log file for script if requested (-l command line option)
    # --------------------------------------------------------------------------------
#    start_time = time()  # record start time
    args = getparser()  # initialize arguments parser

    print('Initializing SRLite Regression script with the following parameters')
    print(f'Date: {args.doi}')
    print(f'Bounding Box:    {args.bbox}')
    print(f'Model Image:    {args.model_image}')
    print(f'Initial Bands (model):    {args.bands_model}')
    print(f'Low Res Image:    {args.low_res_image}')
    print(f'High Res Image:    {args.high_res_image}')
    print(f'Initial Bands (linear/hr data):    {args.bands_data}')
    print(f'Model:    {args.model}')
    print(f'Regression:    {args.regression}')
    print(f'Output Directory: {args.outdir}')
    print(f'Log: {args.logbool}')

    # Initialize log file
    os.system(f'mkdir -p {args.outdir}')  # create output dir
    if args.logbool:  # if command line option -l was given
        # logfile = create_logfile(args, logdir=args.outdir)
        # TODO: make sure logging works without having to specify it
        create_logfile(args, logdir=args.outdir)  # create logfile for std
    print("Command line executed: ", sys.argv)  # saving command into log file

    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time

    # Debug levels:  0-no debug, 2-visualization, 3-detailed diagnostics
    debug_level = 0

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

    # Temporary input - Need to pull from arg list
    bandNamePairList = list([
        ['blue_ccdc', 'BAND-B'],
        ['green_ccdc', 'BAND-G'],
        ['red_ccdc', 'BAND-R'],
        ['nir_ccdc', 'BAND-N']])

    if (debug_level >= 2):
        print(sys.path)
        print(osgeo.gdal.VersionInfo())

    evhrdir = args.high_res_image
#    evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Yukon_Delta/5-toas"
    # evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Senegal/5-toas"
    # evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Fairbanks/5-toas"
    # evhrdir = "/att/nobackup/gtamkin/dev/srlite/input/TOA_v2/Siberia/5-toas"

    ccdcdir = args.model_image
    #ccdcdir = "/home/gtamkin/nobackup/dev/srlite/input/CCDC_v2"

    outpath = args.outdir
    # outpath = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/03012022/Yukon_Delta"
    # outpath = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/03012022/Senegal"
    # outpath = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/03012022/Fairbanks"
    # outpath = "/att/nobackup/gtamkin/dev/srlite/output/big-batch/03012022/Siberia"
    bandNamePairList2 = list(args.bands_model)

    for r_fn_evhr in Path(evhrdir).glob("*.tif"):
        prefix = str(r_fn_evhr).rsplit("/", 1)
        name = str(prefix[1]).split("-toa.tif", 1)
        r_fn_ccdc = os.path.join(ccdcdir + '/' + name[0] + '-ccdc.tif')
        print('\n Processing files: ', r_fn_evhr, r_fn_ccdc)

        fn_list = [str(r_fn_ccdc), str(r_fn_evhr)]

        pl.trace('\nCCDC file=' + str(r_fn_ccdc))
        if (debug_level >= 3):
            getProjection(str(r_fn_ccdc), "CCDC Combo Plot", pl)

        pl.trace('\nEVHR file=' + str(r_fn_evhr))
        if (debug_level >= 3):
            getProjection(str(r_fn_evhr), "EVHR Combo Plot", pl)

        bandPairIndicesList = validateBands(bandNamePairList, fn_list, pl)

        warp_ds_list, warp_ma_list = getIntersection(fn_list)
        pl.trace('\n CCDC shape=' + str(warp_ma_list[0].shape) +
                 ' EVHR shape=' + str(warp_ma_list[1].shape))

        pl.trace('\n Process Bands ....')
        sr_prediction_list = processBands(warp_ds_list, bandPairIndicesList, fn_list,
                                          bandNamePairList, override, pl)

        pl.trace('\n Create Image....')
        outputname = createImage(str(r_fn_evhr), len(bandPairIndicesList), sr_prediction_list, name[0],
                                 bandNamePairList, outpath, pl)

        # Use gdalwarp to create Cloud-optimized Geotiff (COG)
        cogname = outputname.replace("-precog.tif", ".tif")
        command = 'gdalwarp -of cog ' + outputname + ' ' + cogname
        SystemCommand(command)
        if os.path.exists(outputname):
            os.remove(outputname)

        print("\nElapsed Time: " + cogname + ': ',
              (time.time() - start_time) / 60.0)  # time in min
#        break;

    print("\nTotal Elapsed Time for: " + evhrdir + '/*.tif: ',
          (time.time() - start_time) / 60.0)  # time in min

    exit()
    # --------------------------------------------------------------------------------
    # 1) Get the CCDC raster and edit the input CCDC projection 
    #   a.	get_ccdc(date, bounding_box) 
    #   b.	edit_input(ccdc_image, relevant params to adjust) ← adjustment to specify nodata value
    #       and correct projection definition 
    # --------------------------------------------------------------------------------
    # doi, high_res_image_bbox = raster_obj.extract_extents(args.high_res_image, args)
    # raw_ccdc_image = raster_obj.get_ccdc_image(doi)
    # edited_ccdc_image = raster_obj.edit_image(raw_ccdc_image,
    #                                           nodata_value=raster_obj._targetNodata, # enforce no data value
    #                                           srs=raster_obj._targetSRS, # override srs to enforce latitude corrections
    #                                           xres=None,
    #                                           yres=None)
    # --------------------------------------------------------------------------------
    # 2. Run EVHR to get 2m toa 
    # --------------------------------------------------------------------------------
    #   a.	extract bounding_box and date
    #   b.	edit_input(evhr_image, relevant params to adjust) ← adjustment to specify nodata value 
    # raw_evhr_image = raster_obj.get_evhr_image(doi)
    # edited_evhr_image = raster_obj.warp_image(raw_evhr_image,
    #                                            bbox=None,
    #                                            nodata_value=None,
    #                                            srs=raster_obj._targetSRS,
    #                                            xres=raster_obj.model_xres,
    #                                            yres = raster_obj.model_yres,
    #                                            resampling= raster_obj._targetResampling,
    #                                            overwrite=hasattr(args, 'force_overwrite'))

    # 3) Warp, Model, Write output 
    #       a.	warp_inputs(ccdc_image, evhr_image): 
    #           import pygeotools
    # fn_list = [edited_ccdc_image, edited_evhr_image]

    #           # Warp CCDC and EVHR to bounding_box and 30m CCDC grid
    #           ma_list = warplib.memwarp_multi_fn(fn_list evhr_image, extent='intersection', res=30,
    #               t_srs=ccdc_image, r='cubic' 'average', return ma=True)
#    fn_list, warp_ma_list = raster_obj.get_intersection(fn_list)
#     raster_obj.coefficients = raster_obj.get_intersection(fn_list)

    # clip original ccdc based on intersection
    # intersected_evhr_image = fn_list[1]
    # doi2, bbox2 = raster_obj.extract_extents(intersected_evhr_image, args)
    # intersected_ccdc_image = raster_obj.warp_image(edited_ccdc_image,
    #                                           bbox=bbox2,
    #                                           nodata_value=None,
    #                                           srs=raster_obj._targetSRS,
    #                                           xres=raster_obj.model_xres,
    #                                           yres = raster_obj.model_yres,
    #                                           resampling= raster_obj._targetResampling,
    #                                           overwrite=hasattr(args, 'force_overwrite'))
    # fn_list[0] = intersected_ccdc_image

    #       b. Build model
    #           Model = some_regression_model(ma_list[0], ma_list[1])
#    raster_obj.coefficients = raster_obj.build_stats_model(fn_list)
#    raster_obj.coefficients = raster_obj.build_model(warp_ma_list)

    #       c. Apply model back to input EVHR
    #           out_sr = apply_model(model, evhr_image)
    # srliteFn = raster_obj.apply_model(
    #     raster_obj.high_res_image, raster_obj.coefficients, args)
#    srliteFn = raster_obj.apply_model(raw_evhr_image, coefficients, args)

    #       d. Write out SR COG
    #           out_sr.to_file(“filename.tif”, type=’COG’)
    #
    # print("Elapsed Time: " + srliteFn + ': ',
    #       (time() - start_time) / 60.0)  # time in min


if __name__ == "__main__":
    main()
