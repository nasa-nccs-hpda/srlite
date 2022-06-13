"""
Purpose: Build and apply regression model for coefficient identification of raster data using
         low resolution data (~30m) and TARGET inputs. Apply the coefficients to high resolution data (~2m)
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
import time  # tracking time
import argparse  # system libraries
from pathlib import Path

from srlite.model.Context import Context
from srlite.model.RasterLib import RasterLib

from pygeotools.lib import warplib, iolib
import rasterio

########################################
# Point to local pygeotools (not in ilab-kernel by default)
########################################
#sys.path.append('/home/gtamkin/.local/lib/python3.9/site-packages')
#sys.path.append('/adapt/nobackup/people/gtamkin/dev/srlite/src')

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def _main():
    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time
    print(f' tits Command line executed:    {sys.argv}')

    # Initialize context
    contextClazz = Context()
    context = contextClazz.getDict()

    # Get handles to plot and raster classes
    plotLib = contextClazz.getPlotLib()
    rasterLib = RasterLib(int(context[Context.DEBUG_LEVEL]), plotLib)

    for context[Context.FN_TOA] in sorted(Path(context[Context.DIR_TOA]).glob("*.tif")):

        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)
            rasterLib.getAttributeSnapshot(context)

            #  Downscale EVHR TOA from 2m to 30m - suffix root name with '-toa-30m.tif')
            context[Context.FN_SRC] = str(context[Context.FN_TOA])
            context[Context.FN_DEST] = str(context[Context.FN_TOA_DOWNSCALE])
            fileExists = (os.path.exists(context[Context.FN_TOA_DOWNSCALE]))
            if fileExists and (eval(context[Context.CLEAN_FLAG])):
                rasterLib.removeFile(context[Context.FN_TOA_DOWNSCALE], context[Context.CLEAN_FLAG])
            fileExists = (os.path.exists(context[Context.FN_TOA_DOWNSCALE]))
            if not fileExists:
                rasterLib.translate(context)
            rasterLib.getAttributes(str(context[Context.FN_TOA_DOWNSCALE]),
                                    "TOA Downscale Combo Plot")

            #  Warp cloudmask to attributes of EVHR - suffix root name with '-toa_pred_warp.tif')
            context[Context.FN_SRC] = str(context[Context.FN_CLOUDMASK])
            context[Context.FN_DEST] = str(context[Context.FN_CLOUDMASK_DOWNSCALE])
            context[Context.TARGET_ATTR] = str(context[Context.FN_TOA])
            rasterLib.translate(context)
            rasterLib.getAttributes(str(context[Context.FN_CLOUDMASK_DOWNSCALE]), "Cloudmask Warp Combo Plot")

            # Validate that input band name pairs exist in EVHR & CCDC files
            context[Context.FN_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA])]
            context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

            # Get the common pixel intersection values of the EVHR & CCDC files
            context[Context.DS_LIST], context[Context.MA_LIST] = rasterLib.getIntersection(context)
#            context[Context.DS_LIST], context[Context.MA_LIST] = rasterLib.getIntersection(context[Context.FN_LIST])

            # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
            context[Context.PRED_LIST] = rasterLib.performRegression(context)

            # Create COG image from stack of processed bands
            context[Context.FN_SRC] = str(context[Context.FN_TOA])
            context[Context.FN_SUFFIX] = str(Context.FN_SRLITE_NONCOG_SUFFIX)
            context[Context.BAND_NUM] = len(list(context[Context.LIST_TOA_BANDS]))
            context[Context.BAND_DESCRIPTION_LIST] = list(context[Context.LIST_TOA_BANDS])
            context[Context.COG_FLAG] = True
#            context[Context.TARGET_DTYPE] = ccdc_dataset_profile['dtype']
            context[Context.TARGET_NODATA_VALUE] = int(Context.DEFAULT_NODATA_VALUE)
            context[Context.FN_COG] = rasterLib.createImage(context)

        except FileNotFoundError as exc:
            print(exc);
        except BaseException as err:
            print('Run abended.  Error: ', err)
            sys.exit(1)


def main():

    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time
    print(f'Command line executed:    {sys.argv}')

    # Initialize context
    contextClazz = Context()
    context = contextClazz.getDict()

    # Get handles to plot and raster classes
    plotLib = contextClazz.getPlotLib()
    rasterLib = RasterLib(int(context[Context.DEBUG_LEVEL]), plotLib)

    # Define order indices for list processing
    context[Context.LIST_INDEX_TOA] = 0
    context[Context.LIST_INDEX_TARGET] = 1
    context[Context.LIST_INDEX_CLOUDMASK] = -1

    toa_filter = '*' + context[Context.FN_TOA_SUFFIX]
    for context[Context.FN_TOA] in (Path(context[Context.DIR_TOA]).glob(toa_filter)):
#        for context[Context.FN_TOA] in sorted(Path(context[Context.DIR_TOA]).glob(toa_filter)):
        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

            # Remove existing SR-Lite output if clean_flag is activated
            fileExists = (os.path.exists(context[Context.FN_COG]))
            if fileExists and (eval(context[Context.CLEAN_FLAG])):
                rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if not fileExists:

                # Capture input attributes - then align all artifacts to EVHR TOA projection
                rasterLib.getAttributeSnapshot(context)

                #  Downscale EVHR TOA from 2m to 30m - suffix root name with '-toa-30m.tif')
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_TOA_DOWNSCALE])
                fileExists = (os.path.exists(context[Context.FN_TOA_DOWNSCALE]))
                if fileExists and (eval(context[Context.CLEAN_FLAG])):
                        rasterLib.removeFile(context[Context.FN_TOA_DOWNSCALE], context[Context.CLEAN_FLAG])
                fileExists = (os.path.exists(context[Context.FN_TOA_DOWNSCALE]))
                if not fileExists:
                    rasterLib.translate(context)
                rasterLib.getAttributes(str(context[Context.FN_TOA_DOWNSCALE]),
                                        "TOA Downscale Combo Plot")

                # Retrieve target attributes from downscaled TOA
                rasterLib.setTargetAttributes(context, context[Context.FN_TOA_DOWNSCALE])

                #  Warp TARGET to attributes of EVHR - suffix root name with '-toa_pred_warp.tif')
                context[Context.FN_SRC] = str(context[Context.FN_TARGET])
                context[Context.FN_DEST] = str(context[Context.FN_TARGET_DOWNSCALE])
                rasterLib.translate(context)
                rasterLib.getAttributes(str(context[Context.FN_TARGET_DOWNSCALE]), "Target Warp Combo Plot")

                #  Warp cloudmask to attributes of EVHR TOA Downscale, translate not warp() to skip averaging)
                if (eval(context[Context.CLOUD_MASK_FLAG])):
                    context[Context.FN_SRC] = str(context[Context.FN_CLOUDMASK])
                    context[Context.FN_DEST] = str(context[Context.FN_CLOUDMASK_DOWNSCALE])
                    rasterLib.translate(context)
                    rasterLib.getAttributes(str(context[Context.FN_CLOUDMASK_DOWNSCALE]), "Cloudmask Warp Combo Plot")
                    context[Context.LIST_INDEX_CLOUDMASK] = 2

                # Validate that input band name pairs exist in EVHR & CCDC files
                context[Context.FN_LIST] = [str(context[Context.FN_TOA]), str(context[Context.FN_TARGET])]
                context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                # Reproject all other inputs to TOA to ensure equal number of samples
                context[Context.FN_LIST] = [str(context[Context.FN_TOA_DOWNSCALE]), str(context[Context.FN_TARGET_DOWNSCALE])]
 #                                           str(context[Context.FN_CLOUDMASK_DOWNSCALE])]
                if (eval(context[Context.CLOUD_MASK_FLAG])):
                    context[Context.FN_LIST].append(str(context[Context.FN_CLOUDMASK_DOWNSCALE]))
                context[Context.DS_LIST], context[Context.MA_LIST] = rasterLib.getReprojection(context)

                # Save reprojected masked array for target and cloudmask
                context[Context.DS_TOA_DOWNSCALE] = \
                    context[Context.DS_LIST][context[Context.LIST_INDEX_TOA]]
                context[Context.DS_TARGET_DOWNSCALE] = \
                    context[Context.DS_LIST][context[Context.LIST_INDEX_TARGET]]
                context[Context.DS_CLOUDMASK_DOWNSCALE] = \
                    context[Context.DS_LIST][context[Context.LIST_INDEX_CLOUDMASK]]
                context[Context.MA_CLOUDMASK_DOWNSCALE] = \
                    context[Context.MA_LIST][context[Context.LIST_INDEX_CLOUDMASK]]

                # Get the common pixel intersection values of the EVHR & CCDC files (not cloudmask)
                context[Context.FN_INTERSECTION_LIST] = [str(context[Context.FN_TOA_DOWNSCALE]),
                                            str(context[Context.FN_TARGET_DOWNSCALE])]
                intersectedListDs, intersectedListMa = rasterLib.getIntersection(context)

                # Amend reprojected arrays with intersected arrays for TOA and TARGET
                context[Context.DS_LIST][context[Context.LIST_INDEX_TOA]]  = \
                    intersectedListDs[ context[Context.LIST_INDEX_TOA] ]
                context[Context.DS_LIST][context[Context.LIST_INDEX_TARGET]]  = \
                    intersectedListDs[ context[Context.LIST_INDEX_TARGET] ]
                context[Context.MA_LIST][ context[Context.LIST_INDEX_TOA] ] = \
                    intersectedListMa[ context[Context.LIST_INDEX_TOA] ]
                context[Context.MA_LIST][context[Context.LIST_INDEX_TARGET]] = \
                    intersectedListMa[context[Context.LIST_INDEX_TARGET]]

                # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                context[Context.PRED_LIST] = rasterLib.performRegression(context)

                # Create COG image from stack of processed bands
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_SUFFIX] = str(Context.FN_SRLITE_NONCOG_SUFFIX)
                context[Context.BAND_NUM] = len(list(context[Context.LIST_TOA_BANDS]))
                context[Context.BAND_DESCRIPTION_LIST] = list(context[Context.LIST_TOA_BANDS])
                context[Context.COG_FLAG] = True
                #            context[Context.TARGET_DTYPE] = ccdc_dataset_profile['dtype']
                context[Context.TARGET_NODATA_VALUE] = int(Context.DEFAULT_NODATA_VALUE)

                context[Context.FN_DEST] = str(context[Context.FN_COG])
                context[Context.FN_COG] = rasterLib.createImage(context)

        except FileNotFoundError as exc:
            print('File Not Found - Error details: ', exc)
        except BaseException as err:
            print('Run abended - Error details: ', err)
#        break;

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT])  + ': ',
           (time.time() - start_time) / 60.0)  # time in min

def mainARD():

    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time
    print(f'Command line executed:    {sys.argv}')

    # Initialize context
    contextClazz = Context()
    context = contextClazz.getDict()

    # Get handles to plot and raster classes
    plotLib = contextClazz.getPlotLib()
    rasterLib = RasterLib(int(context[Context.DEBUG_LEVEL]), plotLib)

    toa_filter = '*' + [Context.FN_TOA_SUFFIX]
#    for context[Context.FN_TOA] in sorted(Path(context[Context.DIR_TOA]).glob("*.tif")):
    for context[Context.FN_TOA] in (Path(context[Context.DIR_TOA]).glob(toa_filter)):
        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

            # Remove existing SR-Lite output if clean_flag is activated
            fileExists = (os.path.exists(context[Context.FN_COG]))
            if fileExists and (eval(context[Context.CLEAN_FLAG])):
                rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # # Get snapshot of attributes of EVHR, CCDC, and Cloudmask tifs and create plot")
            # toaXform = rasterLib.getAttributes(str(context[Context.FN_TOA]), "EVHR Combo Plot")
            # ccdcXform = rasterLib.getAttributes(str(context[Context.FN_TARGET]), "CCDC Combo Plot")
            # cloudXform = rasterLib.getAttributes(str(context[Context.FN_CLOUDMASK]), "Cloudmask Combo Plot")
            #
            # (xMin, xMax, yMin, yMax) = toaXform.GetEnvelope()
            # rasterLib._plot_lib.trace("Extents = ({}, {}, {}, {})".format(toaXform.GetEnvelope()))
            # print[xMin, xMax, yMin, yMax]
            #
            # Proceed if SR-Lite output does not exist
            if not fileExists:

                # Capture input attributes - align all artifacts to EVHR projection
                rasterLib.getAttributeSnapshot(context)

                #  Downscale EVHR TOA from 2m to 30m - suffix root name with '-toa-30m.tif')
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_TOA_DOWNSCALE])
                fileExists = (os.path.exists(context[Context.FN_TOA_DOWNSCALE]))
                if fileExists and (eval(context[Context.CLEAN_FLAG])):
                        rasterLib.removeFile(context[Context.FN_TOA_DOWNSCALE], context[Context.CLEAN_FLAG])
                fileExists = (os.path.exists(context[Context.FN_TOA_DOWNSCALE]))
                if not fileExists:
                    rasterLib.translate(context)
                rasterLib.getAttributes(str(context[Context.FN_TOA_DOWNSCALE]),
                                        "TOA Downscale Combo Plot")

                # Retrieve target attributes from downscaled TOA
                rasterLib.setTargetAttributes(context, context[Context.FN_TOA_DOWNSCALE])

                # Validate that input band name pairs exist in EVHR & TARGET files
                context[Context.FN_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA_DOWNSCALE])]
                context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)
                context[Context.FN_LIST] = [str(context[Context.FN_TOA_DOWNSCALE]), str(context[Context.FN_TARGET])]

                # Warp TARGET/Landsat to downscaled TOA 30m
                with rasterio.open(context[Context.FN_TOA_DOWNSCALE], "r") as ccdc_dataset:
                    out_meta = ccdc_dataset.meta.copy()
                    n_bands = ccdc_dataset.count
                    ccdc_dataset_profile = ccdc_dataset.profile
                print(f"# of TARGET bands: {n_bands}")
                print(ccdc_dataset_profile)

                with rasterio.open(context[Context.FN_TARGET], "r") as lard_dataset:
                    n_bands_lard = lard_dataset.count
                    lard_dataset_profile = lard_dataset.profile
                    lard_dataset_desc = lard_dataset.descriptions
                print(f"# of LARD bands: {n_bands_lard}")
                print(f"Descriptions of LARD bands: {lard_dataset_desc}")
                print(lard_dataset_profile)

                fn_list = [context[Context.FN_TOA_DOWNSCALE], context[Context.FN_TARGET]]
                # warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first',
                #                                         r='average')
                warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='first', t_srs='first', r='cubic')
                landsat_ds = warp_ds_list[1]
                band_list = []

                for bandnum in range(1, n_bands_lard + 1):
                    # Warp LARD to TARGET
                    landsat_ard_ma = iolib.ds_getma(landsat_ds, bnum=bandnum)
                    band_list.append(landsat_ard_ma)

                context[Context.PRED_LIST] = band_list
                context[Context.FN_SRC] = str(context[Context.FN_TOA_DOWNSCALE])
                context[Context.FN_SUFFIX] = str(Context.FN_TARGET_DOWNSCALE_SUFFIX)
                context[Context.BAND_NUM] = n_bands_lard
                context[Context.BAND_DESCRIPTION_LIST] = lard_dataset_desc
                context[Context.COG_FLAG] = False
                context[Context.TARGET_DTYPE] = lard_dataset_profile['dtype']
                context[Context.TARGET_NODATA_VALUE] = None
                context[Context.FN_TARGET_DOWNSCALE] = rasterLib.createImage(context)

                #  Warp cloudmask to attributes of EVHR - suffix root name with '-toa_pred_warp.tif')
                context[Context.FN_SRC] = str(context[Context.FN_CLOUDMASK])
                context[Context.FN_DEST] = str(context[Context.FN_CLOUDMASK_DOWNSCALE])
                fileExists = (os.path.exists(context[Context.FN_CLOUDMASK_DOWNSCALE]))
                if fileExists and (eval(context[Context.CLEAN_FLAG])):
                        rasterLib.removeFile(context[Context.FN_CLOUDMASK_DOWNSCALE], context[Context.CLEAN_FLAG])
                fileExists = (os.path.exists(context[Context.FN_CLOUDMASK_DOWNSCALE]))
                if not fileExists:
                    rasterLib.translate(context)
                rasterLib.getAttributes(str(context[Context.FN_CLOUDMASK_DOWNSCALE]),
                                        "Cloudmask Downscale Combo Plot")

                # Get the common pixel intersection values of the EVHR & TARGET files
                context[Context.FN_LIST] = [context[Context.FN_TOA_DOWNSCALE], context[Context.FN_TARGET_DOWNSCALE]]
                context[Context.DS_LIST], context[Context.MA_LIST] = rasterLib.getIntersection(context)

                # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                context[Context.PRED_LIST] = rasterLib.performRegression(context)

                # Create COG image from stack of processed bands
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_SUFFIX] = str(Context.FN_SRLITE_NONCOG_SUFFIX)
                context[Context.BAND_NUM] = len(list(context[Context.LIST_TOA_BANDS]))
                context[Context.BAND_DESCRIPTION_LIST] = list(context[Context.LIST_TOA_BANDS])
                context[Context.COG_FLAG] = True
                context[Context.TARGET_DTYPE] = ccdc_dataset_profile['dtype']
                context[Context.TARGET_NODATA_VALUE] = int(Context.DEFAULT_NODATA_VALUE)
                context[Context.FN_COG] = rasterLib.createImage(context)

                break;

        except FileNotFoundError as exc:
            print('File Not Found - Error details: ', exc)
        except BaseException as err:
            print('Run abended - Error details: ', err)
            sys.exit(1)

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT])  + ': ',
           (time.time() - start_time) / 60.0)  # time in min

if __name__ == "__main__":
    from unittest.mock import patch

    REGION = 'RailroadValley'
    #    REGION = 'Fairbanks'
    #    OUTPUTDIR = f'/adapt/nobackup/people/iluser/projects/srlite/output/LANDSAT_v1/05312022/{REGION}/CloudAndQFMask/'
    #    OUTPUTDIR = f'/adapt/nobackup/people/iluser/projects/srlite/output/LANDSAT_v1/05292022/{REGION}/QFMaskOnly/'
    OUTPUTDIR = f'/adapt/nobackup/people/iluser/projects/srlite/output/CCDC_v2/06032022/{REGION}/CloudMaskOnly/'
    #   OUTPUTDIR = f'/adapt/nobackup/people/iluser/projects/srlite/output/LANDSAT_v1/06012022/{REGION}/CCDCAndCloudMaskOnly/'

    start_time = time.time()  # record start time

    # If not arguments specified, use the defaults
    numParms = len(sys.argv) - 1
    if (numParms  == 0):

        with patch("sys.argv",

            ["SrliteWorkflowCommandLineView.py", \
                "-toa_dir", f'/adapt/nobackup/people/gtamkin/dev/srlite/input/Yukon_Delta/toa',
                "-target_dir", "/adapt/nobackup/people/iluser/projects/srlite/input/CCDC_v2",
                "-cloudmask_dir", f'/gpfsm/ccds01/nobackup/people/gtamkin/dev/srlite/input/Yukon_Delta/cloud',
                "-bandpairs","[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]",
            #       "-bandpairs", "[['Layer_1', 'BAND-B'], ['Layer_2', 'BAND-G'], ['Layer_3', 'BAND-R'], ['Layer_4', 'BAND-N']]",
                "-output_dir", f"/adapt/nobackup/people/gtamkin/dev/srlite/output/srlite-0.9.6-06040022-cloudmask/060322/Yukon_Delta",
            #    "--warp_dir", f"{OUTPUTDIR}/warp",
                "--debug", "1",
                "--regressor", "robust",
                "--clean",
            #        "--algorithm", "landsat",
                "--storage", "memory",
                "--cloudmask",
             #   "--qfmask",
             #   "--qfmasklist","0,3,4",
            #         "--thmask",
                 #        "--threshold_range", "-100,2000"
            ]):

            ##############################################
            # main() using default application parameters
            ##############################################
            print(f'Default application parameters: {sys.argv}')
            main()
    else:

        ##############################################
        # main() using command line parameters
        ##############################################
        print(f'Command line executed:    {sys.argv}')
        main()



