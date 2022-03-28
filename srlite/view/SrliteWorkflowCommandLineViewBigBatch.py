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
from time import time  # tracking time
import argparse  # system libraries
from srlite.model.xrasterlib.SurfaceReflectance import SurfaceReflectance


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


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():

    # Design (pseudo code):
    #
    # 0) Prepare for run
    #
    # For each site: 
    #
    # 1) Run EVHR to get 2m toa 
	#   a.	extract bounding_box and date 
	#   b.	edit_input(evhr_image, relevant params to adjust) ← adjustment to specify nodata value 
    #
	# 2) Get the CCDC raster and edit the input CCDC projection 
	#   a.	get_ccdc(bounding_box, date) 
	#   b.	edit_input(ccdc_image, relevant params to adjust) ← adjustment to specify nodata value
    #       and correct projection definition 
    #
	# 3) Warp, Model, Write output 
	#       a.	warp_inputs(ccdc_image, evhr_image): 
    #           import pygeotools
    #          fn_list = [ccdc_image, evhr_image]
    #
    #           # Warp CCDC and EVHR to bounding_box and 30m CCDC grid
    #           ma_list = warplib.memwarp_multi_fn(fn_list evhr_image, extent='intersection', res=30,
    #               t_srs=ccdc_image, r='cubic' 'average', return ma=True)
    #
    #       b. Build model
    #           Model = some_regression_model(ma_list[0], ma_list[1])
    #
    #       c. Apply model back to input EVHR
    #           out_sr = apply_model(model, evhr_image)
    #
    #       d. Write out SR COG
    #           out_sr.to_file(“filename.tif”, type=’COG’)
    #

    # --------------------------------------------------------------------------------
    # 0. Prepare for run - set log file for script if requested (-l command line option)
    # --------------------------------------------------------------------------------
    start_time = time()  # record start time
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

    # Initialize raster object that hosts algorithmic operations
    raster_obj = SurfaceReflectance(
        model_image=args.model_image,
        low_res_image=args.low_res_image,
        high_res_image=args.high_res_image,
        outdir=args.outdir
    )

    # --------------------------------------------------------------------------------
    # 1) Get the CCDC raster and edit the input CCDC projection 
    #   a.	get_ccdc(date, bounding_box) 
    #   b.	edit_input(ccdc_image, relevant params to adjust) ← adjustment to specify nodata value
    #       and correct projection definition 
    # --------------------------------------------------------------------------------
    doi, high_res_image_bbox = raster_obj.extract_extents(args.high_res_image, args)
    raw_ccdc_image = raster_obj.get_ccdc_image(doi)
    edited_ccdc_image = raster_obj.edit_image(raw_ccdc_image,
                                              nodata_value=raster_obj._targetNodata, # enforce no data value
                                              srs=raster_obj._targetSRS, # override srs to enforce latitude corrections
                                              xres=None,
                                              yres=None)
    # --------------------------------------------------------------------------------
    # 2. Run EVHR to get 2m toa 
    # --------------------------------------------------------------------------------
    #   a.	extract bounding_box and date 
    #   b.	edit_input(evhr_image, relevant params to adjust) ← adjustment to specify nodata value 
    raw_evhr_image = raster_obj.get_evhr_image(doi)
    edited_evhr_image = raster_obj.warp_image(raw_evhr_image,
                                               bbox=None,
                                               nodata_value=None,
                                               srs=raster_obj._targetSRS,
                                               xres=raster_obj.model_xres,
                                               yres = raster_obj.model_yres,
                                               resampling= raster_obj._targetResampling,
                                               overwrite=hasattr(args, 'force_overwrite'))

    # 3) Warp, Model, Write output 
    #       a.	warp_inputs(ccdc_image, evhr_image): 
    #           import pygeotools
    fn_list = [edited_ccdc_image, edited_evhr_image]

    #           # Warp CCDC and EVHR to bounding_box and 30m CCDC grid
    #           ma_list = warplib.memwarp_multi_fn(fn_list evhr_image, extent='intersection', res=30,
    #               t_srs=ccdc_image, r='cubic' 'average', return ma=True)
#    fn_list, warp_ma_list = raster_obj.get_intersection(fn_list)
    raster_obj.coefficients = raster_obj.get_intersection(fn_list)

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
    srliteFn = raster_obj.apply_model(
        raster_obj.high_res_image, raster_obj.coefficients, args)
#    srliteFn = raster_obj.apply_model(raw_evhr_image, coefficients, args)

    #       d. Write out SR COG
    #           out_sr.to_file(“filename.tif”, type=’COG’)
    #
    print("Elapsed Time: " + srliteFn + ': ',
          (time() - start_time) / 60.0)  # time in min


if __name__ == "__main__":
    main()
