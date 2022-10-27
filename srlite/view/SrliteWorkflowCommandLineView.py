"""
Purpose: Build and apply regression model for coefficient identification of raster data using
         low resolution data (~30m) and TARGET inputs. Apply the coefficients to high resolution data (~2m)
         to generate a surface reflectance product (aka, SR-Lite). Usage requirements are referenced in README.

Data Source: This script has been tested with very high-resolution WV data.
             Additional testing will be required to validate the applicability
             of this model for other datasets.

Original Author: Glenn Tamkin, CISTO, Code 602
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
    """
    Main routine for SR-Lite
    """
    ##############################################
    # Default configuration values
    ##############################################
    start_time = time.time()  # record start time
    print('Command line executed:    {sys.argv}')

    # Initialize context
    contextClazz = Context()
    context = contextClazz.getDict()

    # Get handles to plot and raster classes
    plotLib = contextClazz.getPlotLib()
    rasterLib = RasterLib(int(context[Context.DEBUG_LEVEL]), plotLib)

    # Retrieve TOA files in sorted order from the input TOA directory and loop through them
    toa_filter = '*' + context[Context.FN_TOA_SUFFIX]
    toaList = [context[Context.DIR_TOA]]
    if os.path.isdir(Path(context[Context.DIR_TOA])):
        toaList = sorted(Path(context[Context.DIR_TOA]).glob(toa_filter))

    for context[Context.FN_TOA] in toaList:
        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

            # Remove existing SR-Lite output if clean_flag is activated
            rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if not (os.path.exists(context[Context.FN_COG])):

                # Capture input attributes - then align all artifacts to EVHR TOA projection
                rasterLib.getAttributeSnapshot(context)

                # Define order indices for list processing
                context[Context.LIST_INDEX_TARGET] = 0
                context[Context.LIST_INDEX_TOA] = 1
                context[Context.LIST_INDEX_CLOUDMASK] = -1  # increment if cloudmask requested

                # Validate that input band name pairs exist in EVHR & CCDC files
                context[Context.FN_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA])]
                context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                #  Reproject TARGET (CCDC) to attributes of EVHR TOA Downscale  - use 'average' for resampling method
                context[Context.FN_REPROJECTION_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA])]
                context[Context.TARGET_FN] = str(context[Context.FN_TOA])
                context[Context.TARGET_SAMPLING_METHOD] = 'average'
                context[Context.DS_WARP_LIST], context[Context.MA_WARP_LIST] = rasterLib.getReprojection(context)

                #  Reproject cloudmask to attributes of EVHR TOA Downscale  - use 'mode' for resampling method
                if eval(context[Context.CLOUD_MASK_FLAG]):
                    context[Context.FN_REPROJECTION_LIST] = [str(context[Context.FN_CLOUDMASK])]
                    context[Context.TARGET_FN] = str(context[Context.FN_TOA])
                    context[Context.TARGET_SAMPLING_METHOD] = 'mode'
                    context[Context.DS_WARP_CLOUD_LIST], context[
                        Context.MA_WARP_CLOUD_LIST] = rasterLib.getReprojection(context)
                    context[Context.LIST_INDEX_CLOUDMASK] = 2

                # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                context[Context.PRED_LIST], context[Context.METRICS_LIST], context[Context.COMMON_MASK_LIST] = \
                    rasterLib.simulateSurfaceReflectance(context)

                # Create COG image from stack of processed bands
                context[Context.FN_SRC] = str(context[Context.FN_TOA])
                context[Context.FN_DEST] = str(context[Context.FN_COG])
                context[Context.BAND_NUM] = len(list(context[Context.LIST_TOA_BANDS]))
                context[Context.BAND_DESCRIPTION_LIST] = list(context[Context.LIST_TOA_BANDS])
                context[Context.FN_COG] = rasterLib.createImage(context)

                # Generate simulated bands
                if eval(context[Context.BAND8_FLAG]):
                    context[Context.FN_COG_8BAND], context[Context.METRICS_LIST] = \
                        rasterLib.apply_regressor_otf_8band(context,
                                                            context[Context.FN_COG])

#                     # Create 8-band COG image from stack of processed bands
# #                    context[Context.PRED_LIST] = result_weighted_8band
#                     context[Context.FN_DEST] = str(context[Context.FN_COG])
#                     context[Context.FN_SRC] = context[Context.FN_TOA]
#                     # GT - fix hard-coded 8-band handling
#                     context[Context.BAND_NUM] = int(8)
#                     context[Context.BAND_DESCRIPTION_LIST] = \
#                         ['BAND-B', 'BAND-G', 'BAND-R', 'BAND-N', 'BAND-C',
#                          'BAND-Y', 'BAND-RE', 'BAND-N2']
#                     context[Context.FN_COG] = rasterLib.createImage(context)

                # Generate CSV
                rasterLib.generateCSV(context, context[Context.METRICS_LIST])

                # Clean up
                rasterLib.refresh(context)

        except FileNotFoundError as exc:
            print('File Not Found - Error details: ', exc)
        except BaseException as err:
            print('Run abended - Error details: ', err)

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT]) + ': ',
          (time.time() - start_time) / 60.0)  # time in min


if __name__ == "__main__":
    from unittest.mock import patch

    start_time = time.time()  # record start time

    maindir = '/adapt/nobackup/projects/ilab/data/srlite'

    r_fn_ccdc = \
        '/adapt/nobackup/projects/ilab/data/srlite/ccdc/ccdc_v20220620/WV02_20150616_M1BS_103001004351F000-ccdc.tif'
    r_fn_evhr = \
        '/gpfsm/ccds01/nobackup/projects/ilab/data/srlite/toa/Yukon_Delta/WV02_20150616_M1BS_103001004351F000-toa.tif'
    r_fn_cloud = \
        '/gpfsm/ccds01/nobackup/projects/ilab/data/srlite/cloudmask/Yukon_Delta/' + \
        'WV02_20150616_M1BS_103001004351F000-toa.cloudmask.v1.2.tif'

    #
    # region = 'Yukon_Delta'
    # scene = 'WV02_20150616_M1BS_103001004351F000'
    # version = '20220710'
    # OUTDIR = '/adapt/nobackup/people/pmontesa/userfs02/projects/srlite/misc'

    # If not arguments specified, use the defaults
    numParms = len(sys.argv) - 1
    if numParms == 0:

        with patch("sys.argv",

                   ["SrliteWorkflowCommandLineView.py",
                    "-toa_dir", r_fn_evhr,
                    "-target_dir", r_fn_ccdc,
                    "-cloudmask_dir", r_fn_cloud,
                    "-bandpairs",
                    "[['blue_ccdc', 'BAND-B'],['green_ccdc','BAND-G'],['red_ccdc','BAND-R'],['nir_ccdc','BAND-N']]",
                    #                "-bandpairs", "[['BAND-B', 'blue_ccdc'], ['BAND-G', 'green_ccdc'],
                    #                ['BAND-R', 'red_ccdc'], ['BAND-N', 'nir_ccdc']]",
                    "-output_dir", "../../../output/Yukon_Delta/09062022-9.12-rma",
                    "--debug", "1",
                    "--regressor", "rma",
                    "--clean",
                    "--cloudmask",
                    "--pmask",
                    "--csv",
                    ]):
            ##############################################
            # main() using default application parameters
            ##############################################
            print("Default application parameters: {sys.argv}")
            main()
    else:

        ##############################################
        # main() using command line parameters
        ##############################################
        main()
