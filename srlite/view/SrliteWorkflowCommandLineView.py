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
from pathlib import Path

from srlite.model.Context import Context
from srlite.model.RasterLib import RasterLib

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

    # Retrieve TOA files in sorted order from the input TOA directory and loop through them
    toa_filter = '*' + context[Context.FN_TOA_SUFFIX]
    toaList=[context[Context.DIR_TOA]]
    if os.path.isdir(Path(context[Context.DIR_TOA])):
        toaList = sorted(Path(context[Context.DIR_TOA]).glob(toa_filter))

    for context[Context.FN_TOA] in toaList:
        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

             # Remove existing SR-Lite output if clean_flag is activated
            rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if  not (os.path.exists(context[Context.FN_COG])):

                # Capture input attributes - then align all artifacts to EVHR TOA projection
                rasterLib.getAttributeSnapshot(context)

                # Define order indices for list processing
                context[Context.LIST_INDEX_TARGET] = 0
                context[Context.LIST_INDEX_TOA] = 1
                context[Context.LIST_INDEX_CLOUDMASK] = -1  # increment if cloudmask requested

                # Validate that input band name pairs exist in EVHR & CCDC files
                context[Context.FN_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA])]
                context[Context.MA_LIST]  = rasterLib.getMaskedArrays(context)
                context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                # Reproject all other inputs to TOA to ensure equal number of samples
                if (eval(context[Context.CLOUD_MASK_FLAG])):
                    context[Context.FN_LIST].append(str(context[Context.FN_CLOUDMASK]))
                    context[Context.LIST_INDEX_CLOUDMASK] = 2
                context[Context.DS_WARP_LIST], context[Context.MA_WARP_LIST] = rasterLib.getReprojection(context)

                # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                context[Context.PRED_LIST] = rasterLib.performRegression(context)

                # Create COG image from stack of processed bands
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_COG])
                context[Context.FN_COG] = rasterLib.createImage(context)

                # Clean up
 #               rasterLib.refresh(context)

        except FileNotFoundError as exc:
            print('File Not Found - Error details: ', exc)
        except BaseException as err:
            print('Run abended - Error details: ', err)

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT])  + ': ',
           (time.time() - start_time) / 60.0)  # time in min

def main99plus():

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

    # Retrieve TOA files in sorted order from the input TOA directory and loop through them
    toa_filter = '*' + context[Context.FN_TOA_SUFFIX]
    toaList=[context[Context.DIR_TOA]]
    if os.path.isdir(Path(context[Context.DIR_TOA])):
        toaList = sorted(Path(context[Context.DIR_TOA]).glob(toa_filter))

    for context[Context.FN_TOA] in toaList:
        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

             # Remove existing SR-Lite output if clean_flag is activated
            rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if  not (os.path.exists(context[Context.FN_COG])):

                # Capture input attributes - then align all artifacts to EVHR TOA projection
                rasterLib.getAttributeSnapshot(context)

                # Validate that input band name pairs exist in EVHR & CCDC files
                context[Context.FN_LIST] = [str(context[Context.FN_TOA]), str(context[Context.FN_TARGET])]
                context[Context.MA_LIST] = rasterLib.getMaskedArrays(context[Context.FN_LIST])
                context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                #  Downscale inputs to attributes of EVHR TOA Downscale, translate not warp() to skip averaging)
                if (eval(context[Context.CLOUD_MASK_FLAG])):
                    context[Context.FN_LIST].append(str(context[Context.FN_CLOUDMASK]))
                context[Context.DS_WARP_LIST], context[Context.MA_WARP_LIST] = rasterLib.getReprojection(context)

                # If Cloudmask is desired, apply it to the warped arrays.  Otherwise, retrieve existing warped arrays
                if (eval(context[Context.CLOUD_MASK_FLAG])):
                    context[Context.MA_WARP_LIST] = rasterLib.applyEVHRCloudmask(context)

                rasterLib.getStatistics(context)

                # Get a common mask of valid data from the inputs
                context[Context.COMMON_MASK] = rasterLib.getCommonMask(context)

                # Warped masked arrays with common mask applied
                context[Context.MA_WARP_MASKED_LIST] = rasterLib.applyCommonMask(context)

                #We've used the cloudmask to mask out pixels of toa clouds that we dont want to contaminate the model
                #It has served its purpose, and we can remove it from our list
                context[Context.MA_WARP_MASKED_LIST] = context[Context.MA_WARP_MASKED_LIST] [0:2]

                # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                context[Context.PRED_LIST] = rasterLib.performRegression(context)

                # Create COG image from stack of processed bands
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_COG])
                context[Context.FN_COG] = rasterLib.createImage(context)

                # Clean up
                rasterLib.refresh(context)

        except FileNotFoundError as exc:
            print('File Not Found - Error details: ', exc)
        except BaseException as err:
            print('Run abended - Error details: ', err)

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT])  + ': ',
           (time.time() - start_time) / 60.0)  # time in min

def __main99():

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

    # Retrieve TOA files in sorted order from the input TOA directory and loop through them
    toa_filter = '*' + context[Context.FN_TOA_SUFFIX]
    toaList=[context[Context.DIR_TOA]]
    if os.path.isdir(Path(context[Context.DIR_TOA])):
        toaList = sorted(Path(context[Context.DIR_TOA]).glob(toa_filter))

    for context[Context.FN_TOA] in toaList:
        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

             # Remove existing SR-Lite output if clean_flag is activated
            rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if  not (os.path.exists(context[Context.FN_COG])):

                # Capture input attributes - then align all artifacts to EVHR TOA projection
                rasterLib.getAttributeSnapshot(context)

                # Define order indices for list processing
                context[Context.LIST_INDEX_TOA] = 0
                context[Context.LIST_INDEX_TARGET] = 1
                context[Context.LIST_INDEX_CLOUDMASK] = -1  # increment if cloudmask requested

                #  Downscale EVHR TOA from 2m to 30m - suffix root name with '-toa-30m.tif')
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_TOA_DOWNSCALE])
                rasterLib.downscale(context)

                # Retrieve target attributes from downscaled TOA (for memwarp guidance)
                rasterLib.setTargetAttributes(context, context[Context.FN_TOA_DOWNSCALE])

                #  Downscale TARGET to attributes of EVHR
                context[Context.FN_SRC] = str(context[Context.FN_TARGET])
                context[Context.FN_DEST] = str(context[Context.FN_TARGET_DOWNSCALE])
                rasterLib.downscale(context)

                #  Downscale cloudmask to attributes of EVHR TOA Downscale, translate not warp() to skip averaging)
                if (eval(context[Context.CLOUD_MASK_FLAG])):
                    context[Context.FN_SRC] = str(context[Context.FN_CLOUDMASK])
                    context[Context.FN_DEST] = str(context[Context.FN_CLOUDMASK_DOWNSCALE])
                    rasterLib.downscale(context)
                    context[Context.LIST_INDEX_CLOUDMASK] = 2

                # Validate that input band name pairs exist in EVHR & CCDC files
                context[Context.FN_LIST] = [str(context[Context.FN_TOA]), str(context[Context.FN_TARGET])]
                context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                # Reproject all other inputs to TOA to ensure equal number of samples
                context[Context.FN_LIST] = [str(context[Context.FN_TOA_DOWNSCALE]), str(context[Context.FN_TARGET_DOWNSCALE])]
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
                context[Context.DS_INTERSECTION_LIST] = [context[Context.DS_TOA_DOWNSCALE] ,
                                            context[Context.DS_TARGET_DOWNSCALE]]
                intersectedListDs, intersectedListMa = rasterLib.getIntersectionDs(context)

                # Amend reprojected arrays with intersected arrays for TOA and TARGET
                context[Context.DS_LIST][context[Context.LIST_INDEX_TOA]]  = \
                    intersectedListDs[context[Context.LIST_INDEX_TOA] ]
                context[Context.DS_LIST][context[Context.LIST_INDEX_TARGET]]  = \
                    intersectedListDs[context[Context.LIST_INDEX_TARGET] ]
                context[Context.MA_LIST][context[Context.LIST_INDEX_TOA] ] = \
                    intersectedListMa[context[Context.LIST_INDEX_TOA] ]
                context[Context.MA_LIST][context[Context.LIST_INDEX_TARGET]] = \
                    intersectedListMa[context[Context.LIST_INDEX_TARGET]]

                # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                context[Context.PRED_LIST] = rasterLib.performRegression(context)

                # Create COG image from stack of processed bands
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_COG])
                context[Context.FN_COG] = rasterLib.createImage(context)

                # Clean up
                rasterLib.refresh(context)

        except FileNotFoundError as exc:
            print('File Not Found - Error details: ', exc)
        except BaseException as err:
            print('Run abended - Error details: ', err)

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT])  + ': ',
           (time.time() - start_time) / 60.0)  # time in min

if __name__ == "__main__":
    from unittest.mock import patch

    start_time = time.time()  # record start time

    maindir = '/adapt/nobackup/projects/ilab/data/srlite'

    #
    # # Old inputs
    # # r_fn_ccdc = os.path.join(maindir, 'ccdc_20110818.tif')
    # # r_fn_evhr = os.path.join(maindir, 'WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif')

#/adapt/nobackup/people/iluser/projects/srlite/input/CCDC_v2/WV02_20150616_M1BS_103001004351F000-ccdc.tif
#/gpfsm/ccds01/nobackup/people/iluser/projects/srlite/input/TOA_v2/Yukon_Delta/5-toas/WV02_20150616_M1BS_103001004351F000-toa.tif
#/adapt/nobackup/projects/ilab/data/srlite/cloudmask/Yukon_Delta/WV02_20150616_M1BS_103001004351F000-toa.cloudmask.v1.2.tif
    #
    # # New inputs
    # r_fn_ccdc = os.path.join(maindir, 'ccdc/CCDC_ALL/WV02_20180527_M1BS_103001007E5F8400-ccdc.tif')
    # r_fn_evhr = os.path.join(maindir, 'toa/Alaska/WV02_20180527_M1BS_103001007E5F8400-toa.tif')
    # r_fn_cloud = os.path.join(maindir, 'cloudmask/Alaska/WV02_20180527_M1BS_103001007E5F8400-toa.cloudmask.v1.2.tif')
    #
    r_fn_ccdc = '/adapt/nobackup/people/iluser/projects/srlite/input/CCDC_v2/WV02_20150616_M1BS_103001004351F000-ccdc.tif'
    r_fn_evhr = '/gpfsm/ccds01/nobackup/people/iluser/projects/srlite/input/TOA_v2/Yukon_Delta/5-toas/WV02_20150616_M1BS_103001004351F000-toa.tif'
    r_fn_cloud = '/adapt/nobackup/projects/ilab/data/srlite/cloudmask/Yukon_Delta/WV02_20150616_M1BS_103001004351F000-toa.cloudmask.v1.2.tif'
    # fn_list =[r_fn_ccdc, r_fn_evhr, r_fn_cloud]
    #
    # OUTDIR = '/adapt/nobackup/people/pmontesa/userfs02/projects/srlite/misc'
    # If not arguments specified, use the defaults
    numParms = len(sys.argv) - 1
    if (numParms  == 0):

        with patch("sys.argv",

        ["SrliteWorkflowCommandLineView.py", \
                "-toa_dir", r_fn_evhr,
                "-target_dir", r_fn_ccdc,
                "-cloudmask_dir", r_fn_cloud,
                "-bandpairs", "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]",
#                "-bandpairs", "[['BAND-B', 'blue_ccdc'], ['BAND-G', 'green_ccdc'], ['BAND-R', 'red_ccdc'], ['BAND-N', 'nir_ccdc']]",
                "-output_dir", "../../../output/Yukon_Delta/07042022-9.10",
                "--debug", "1",
                "--regressor", "simple",
                "--clean",
                "--cloudmask",
             ]):
        #
            # ["SrliteWorkflowCommandLineView.py", \
            #  "-toa_dir", "../../../input/Fairbanks",
            #  "-target_dir", "../../../input/Fairbanks",
            #  "-cloudmask_dir", "../../../input/Fairbanks",
            #  "-bandpairs",
            #  "[['BAND-B', 'blue_ccdc'], ['BAND-G', 'green_ccdc'], ['BAND-R', 'red_ccdc'], ['BAND-N', 'nir_ccdc']]",
            #  "-output_dir", "../../../output/Fairbanks/07032022-toa-as-reference-test",
            #  "--debug", "1",
            #  "--regressor", "robust",
            #  "--clean",
            #  "--cloudmask",
            #  ]):

            ##############################################
            # main() using default application parameters
            ##############################################
            print(f'Default application parameters: {sys.argv}')
            main()
    else:

        ##############################################
        # main() using command line parameters
        ##############################################
        main()



