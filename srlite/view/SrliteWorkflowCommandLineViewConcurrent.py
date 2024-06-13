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
#sys.path.insert(0,'/explore/nobackup/people/gtamkin/dev/srlite/src')

from srlite.model.Context import Context
from srlite.model.RasterLib import RasterLib
from srlite.model.SrliteWorkflow import SrliteWorkflow

# import pandas as pd
import multiprocessing as multiprocessing

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
    srlWorkflow = SrliteWorkflow(int(context[Context.DEBUG_LEVEL]), plotLib)

    # Retrieve TOA files in sorted order from the input TOA directory and loop through them
    toa_filter = '*' + context[Context.FN_TOA_SUFFIX]
    toaList = [context[Context.DIR_TOA]]
    if os.path.isdir(Path(context[Context.DIR_TOA])):
        toaList = sorted(Path(context[Context.DIR_TOA]).glob(toa_filter))

    errorIndex = 0
    sr_errors_list = []

    num_workers = len(toaList)   
    items = [(toaList[i], contextClazz, context, rasterLib) for i in range(0, num_workers)]
    #for context[Context.FN_TOA] in toaList:
    try:
        if (num_workers > 0):
             # build a non-blocking processing pool map (i.e. async_map)
            from pathos.multiprocessing import ProcessingPool,ThreadingPool
            tmap = ThreadingPool().map
            # amap = ProcessingPool().amap            
            
            # from multiprocessing.pool import Pool
            print('max processes: ', multiprocessing.cpu_count(), ' processes requested from pool: ', num_workers)

            # results = [num_workers]
            # for i in range(num_workers-1):
            for i in range(num_workers):
                print(f'Starting ProcessingPool().tmap() for band pair: {str(toaList[i])}', flush=True)
                tmap(srlWorkflow.processToa, [toaList[i]], [contextClazz], [context], [rasterLib])
                # time.sleep(10)
                print(f'End ProcessingPool().tmap() for band pair: {str(toaList[i])}', flush=True)

        else:
            print('Processing complete.  See:', context[Context.DIR_OUTPUT])
        
    except FileNotFoundError as exc:
        print('File Not Found - Error details: ', exc)
    except BaseException as err:
        print('Run abended - Error details: ', err)
    
    finally:

            # Delete interim noncog files
            rasterLib.purge(context[Context.DIR_OUTPUT], str(Context.FN_SRLITE_NONCOG_SUFFIX))

            # Delete interim warp files
            rasterLib.purge(context[Context.DIR_OUTPUT_WARP], str(Context.FN_WARP_SUFFIX))


    # Generate Error Report
    context[Context.ERROR_LIST] = sr_errors_list
    if eval(context[Context.ERROR_REPORT_FLAG]):
        rasterLib.generateErrorReport(context)

    print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT]) + ': ',
          (time.time() - start_time) / 60.0)  # time in min

if __name__ == "__main__":
    from unittest.mock import patch

    start_time = time.time()  # record start time

    maindir = '/adapt/nobackup/projects/ilab/data/srlite'

    r_fn_ccdc = \
        '/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline/WV02_20150911_M1BS_1030010049148A00-ccdc.tif'
    r_fn_evhr = \
        '/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline/WV02_20150911_M1BS_1030010049148A00-toa.tif'
    r_fn_cloud = \
        '/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline/WV02_20150911_M1BS_1030010049148A00-toa.cloudmask.v1.2.tif'


    # If not arguments specified, use the defaults
    numParms = len(sys.argv) - 1
    if numParms == 0:

        with patch("sys.argv",

                   ["SrliteWorkflowCommandLineView.py",
                    "-toa_dir", r_fn_evhr,
                    "-target_dir", r_fn_ccdc,
                    "-cloudmask_dir", r_fn_cloud,
                    "-bandpairs",
                    #"[['blue_ccdc', 'BAND-B'],['green_ccdc','BAND-G'],['red_ccdc','BAND-R'],['nir_ccdc','BAND-N']]",
                    #                "-bandpairs", "[['BAND-B', 'blue_ccdc'], ['BAND-G', 'green_ccdc'],
                    #                ['BAND-R', 'red_ccdc'], ['BAND-N', 'nir_ccdc']]",
                    "[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N'], ['blue_ccdc', 'BAND-C'], ['green_ccdc', 'BAND-Y'], ['red_ccdc', 'BAND-RE'], ['nir_ccdc', 'BAND-N2']]",
                    "-output_dir", "/explore/nobackup/people/gtamkin/dev/srlite/test/v2_srlite-2.0-rma-baseline/20240305-cantfingbelieveit",
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
