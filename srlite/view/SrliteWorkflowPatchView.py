
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
from srlite.model.SrliteWorkflow import SrliteWorkflow
from srlite.model.Context import Context
from unittest.mock import patch

def main():
    """
    Main routine for SR-Lite Python API orchestrator
    """
    logger=None
    debug = "1"

    ##############################################
    # Configuration values:
    #   - Directories must be specified for each invocation.
    #   - Commonly modified arguments are specified in patch(sys.argv) below; however, default values are generally sensible for most runs.
    ##############################################
    
    # BASELINE TOA 
    toa_file = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline/WV02_20150911_M1BS_1030010049148A00-toa.tif"
    # toa_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    target_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    cloudmask_dir = "/panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline"
    output_dir = "/explore/nobackup/people/gtamkin/dev/srlite/test/srlite-GI#27_Add_API_class/20241008-band8-cloudmask-API"

    # Create args to simulate CLI 
    with patch("sys.argv",

                ["SrliteWorkflowAPIView.py",
                "-toa_dir", toa_file,               # input TOA path (required - specify above)
                "-target_dir", target_dir,          # input CCDC path (required - specify above)
                "-cloudmask_dir", cloudmask_dir,    # input Cloudmask path (required - specify above)
                "-output_dir", output_dir,          # Output path (required - specify above)
                "--debug", "1",                     # debug level - lower to reduce tracing
                "--clean",                          # Remove all trace of previous TOA processing (omit if undesired)
                "--cloudmask",                      # Enable cloudmasking (omit if undesired)
                "--toa_suffix", "toa.tif",          # Specify TOA file suffix (default = toa.tif')
                "--target_suffix", "ccdc.tif",       # Specify TARGET file suffix (default = ccdc.tif')
                "--cloudmask_suffix", "toa.cloudmask.v1.2.tif",      # Specify CLOUDMASK file suffix (default = toa.cloudmask.v1.2.tif')
                "--scenes_in_file", "/explore/nobackup/projects/ilab/projects/vhr-toolkit/testing/evhr/toa_clv_test_alaska_cloud.csv",
                ]):

        # Initialize workflow - See the following link for parameter descriptions:
        #   https://github.com/nasa-nccs-hpda/srlite/blob/main/srlite/model/SrliteWorkflow.py
        #
        # USAGE:  toa_dir can point to a directory OR a specific file.  If a directory is provided,
        # call processToas() with no override parameters.  To process specific TOAs, send a fully
        # qualified path to processToas(toa).
        #
        srlWorkflow = SrliteWorkflow(Context(output_dir, debug), logger)

        # Example runs:

        #Process all TOAs in the toa_dir directory
        srlWorkflow.processToas()

        # Process a specific TOA
        srlWorkflow.processToa(toa_file)

if __name__ == "__main__":
        main()
