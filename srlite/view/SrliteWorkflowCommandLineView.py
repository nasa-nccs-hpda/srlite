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
import ast
import time  # tracking time
import numpy as np
from pygeotools.lib import iolib, malib
from pathlib import Path

from srlite.model.Context import Context
from srlite.model.RasterLib import RasterLib
from sklearn.linear_model import HuberRegressor, LinearRegression

########################################
# Point to local pygeotools (not in ilab-kernel by default)
########################################
sys.path.append('/home/gtamkin/.local/lib/python3.9/site-packages')
sys.path.append('/adapt/nobackup/people/gtamkin/dev/srlite/src')

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
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

    for context[Context.FN_TOA] in sorted(Path(context[Context.DIR_TOA]).glob("*.tif")):
#    for context[Context.FN_TOA] in (Path(context[Context.DIR_TOA]).glob("*.tif")):

        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)
            rasterLib.getAttributeSnapshot(context)

            #  Warp cloudmask to attributes of EVHR - suffix root name with '-toa_pred_warp.tif')
            context[Context.FN_SRC] = str(context[Context.FN_CLOUDMASK])
            context[Context.FN_DEST] = str(context[Context.FN_WARP])
            context[Context.TARGET_ATTR] = str(context[Context.FN_TOA])
            rasterLib.translate(context)
            rasterLib.getAttributes(str(context[Context.FN_WARP]), "Cloudmask Warp Combo Plot")

            # Validate that input band name pairs exist in EVHR & CCDC files
            context[Context.FN_LIST] = [str(context[Context.FN_CCDC]), str(context[Context.FN_TOA])]
            context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

            # Get the common pixel intersection values of the EVHR & CCDC files
            context[Context.DS_LIST], context[Context.MA_LIST] = rasterLib.getIntersection(context)

            # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
            context[Context.PRED_LIST] = rasterLib.performRegression(context)

            # Create COG image from stack of processed bands
            context[Context.FN_COG] = rasterLib.createImage(context)
 
        except FileNotFoundError as exc:
            print(exc);
        except BaseException as err:
            print('Run abended.  Error: ', err)
            sys.exit(1)

#        break;

    print("\nTotal Elapsed Time for " + context[Context.FN_COG] + ': ',
           (time.time() - start_time) / 60.0)  # time in min

if __name__ == "__main__":
    main()
