
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
import sys
import os
import time  # tracking time
from pathlib import Path

from srlite.model.Context import Context
from srlite.model.RasterLib import RasterLib
from srlite.model.SrliteWorkflow import SrliteWorkflow

def main():
    """
    Main routine for SR-Lite
    """
    print('Command line executed:    {sys.argv}')
    
    ##############################################
    # Default configuration values
    ##############################################

    # BIGTIFF TOA
    # toa_dir = "/explore/nobackup/projects/ilab/data/srlite/products/srlite_1.0.1/toa/alaska_2nd_batch/split_4/WV02_20170906_M1BS_103001006F1F5D00-toa.tif"
    # target_dir = "/explore/nobackup/projects/ilab/data/srlite/ccdc/ccdc_20230807_alaska_batch23/alaska"
    # cloudmask_dir = "/explore/nobackup/projects/ilab/data/srlite/products/srlite_1.0.1/cloudmask/alaska_batch_2/split_4"
    # output_dir = "/explore/nobackup/people/gtamkin/dev/srlite/test/srlite-GI#25_Address_Maximum_TIFF_file_size_exceeded/20240929-vscode-API-ilab213"
    # cloudmask_suffix="-toa.cloudmask.tif" 
    
    # BASELINE TOA
    toa_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    target_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    cloudmask_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    output_dir = "/explore/nobackup/people/gtamkin/dev/srlite/test/srlite-2.0-rma-baseline/20240930-api"
    cloudmask_suffix="-toa.cloudmask.v1.2.tif"

    # DEFAULT TOA
    logger=None

    srlWorkflow = SrliteWorkflow(output_dir=output_dir, 
                                 toa_src=toa_dir,
                                 target_dir=target_dir, 
                                 cloudmask_dir=cloudmask_dir,
                                 regressor="rma",
                                 debug=1,
                                 pmask="True",
                                 cloudmask="True",
                                 csv="True",
                                 band8="True",
                                 clean="True",
                                 cloudmask_suffix=cloudmask_suffix, 
                                 target_suffix="-ccdc.tif",
                                 logger=logger)
    print('Command line executed:    {' +str(sys.argv) + '}')
    print('Initial context:    {' +str(srlWorkflow.context) + '}')

    srlWorkflow.processToas()

    # srlWorkflow.processToas(srlWorkflow.context[Context.LIST_TOA_BANDS])

if __name__ == "__main__":
        main()
