
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
import sys
from pathlib import Path

from srlite.model.SrliteWorkflow import SrliteWorkflow

def main():
    """
    Main routine for SR-Lite Python API orchestrator
    """
    
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
    toa_file = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline/WV02_20150911_M1BS_1030010049148A00-toa.tif"
    toa_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    target_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    cloudmask_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    output_dir = "/explore/nobackup/people/gtamkin/dev/srlite/test/srlite-GI#27_Add_API_class/20241009-SrliteWorkflowAPIView"
    cloudmask_suffix="-toa.cloudmask.v1.2.tif"

    # DEFAULT TOA
    logger=None
    # output_dir = None

    # Initialize workflow - See the following link for parameter descriptions:
    #   https://github.com/nasa-nccs-hpda/srlite/blob/main/srlite/model/SrliteWorkflow.py
    #
    # USAGE:  toa_dir can point to a directory OR a specific file.  If a directory is provided,
    # call processToas() with no override parameters.  To process specific TOAs, send a fully
    # qualified path to processToas(toa).
    #
    srlWorkflow = SrliteWorkflow(output_dir=output_dir, 
                                 toa_src=toa_file,
                                 target_dir=target_dir, 
                                 cloudmask_dir=cloudmask_dir,
                                 debug=1,
                                 pmask="True",
                                 cloudmask="True",
                                 csv="True",
                                 band8="True",
                                 clean="True",
                                 cloudmask_suffix=cloudmask_suffix, 
                                 target_suffix="-ccdc.tif",
                                 logger=logger)

    # Example runs:

    # Process all TOAs in the toa_dir directory
    srlWorkflow.processToas()

    # Process a specific TOA
    srlWorkflow.processToa(toa_file)

if __name__ == "__main__":
        main()
