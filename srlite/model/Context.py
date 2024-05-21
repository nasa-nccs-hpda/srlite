#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse  # system libraries
from datetime import datetime
from srlite.model.PlotLib import PlotLib
from pathlib import Path
import csv


# -----------------------------------------------------------------------------
# class Context
#
# This class is a serializable context for orchestration.
# -----------------------------------------------------------------------------
class Context(object):
    # Custom name for current run
    BATCH_NAME = 'batch_name'
    CAT_ID = 'catid'

    # Directories
    DIR_TOA = 'dir_toa'
    DIR_TARGET = 'dir_target'
    DIR_CLOUDMASK = 'dir_cloudmask'
    DIR_OUTPUT = 'dir_out'
    DIR_OUTPUT_CSV = 'dir_out_csv'
    DIR_OUTPUT_ERROR = 'dir_out_err'
    DIR_OUTPUT_WARP = 'dir_out_warp'

    # File names
    FN_DEST = 'fn_dest'
    FN_SRC = 'fn_src'
    FN_LIST = 'fn_list'
    FN_REPROJECTION_LIST = 'fn_reprojection_list'
    DS_LIST = 'ds_list'
    DS_WARP_LIST = 'ds_warp_list'
    DS_INTERSECTION_LIST = 'ds_intersection_list'
    DS_WARP_CLOUD_LIST = 'ds_warp_cloud_list'
    MA_LIST = 'ma_list'
    MA_WARP_LIST = 'ma_warp_list'
    MA_WARP_CLOUD_LIST = 'ma_warp_cloud_list'
    MA_WARP_VALID_LIST = 'ma_warp_valid_list'
    MA_WARP_MASKED_LIST = 'ma_warp_masked_list'
    PRED_LIST = 'pred_list'
    METRICS_LIST = 'metrics_list'
    ERROR_LIST = 'error_list'
    COMMON_MASK_LIST = 'common_mask_list'
    COMMON_MASK = 'common_mask'

    FN_TOA = 'fn_toa'
    FN_TOA_DOWNSCALE = 'fn_toa_downscale'
    DS_TOA_DOWNSCALE = 'ds_toa_downscale'
    MA_TOA_DOWNSCALE = 'ds_toa_downscale'
    FN_TARGET = 'fn_target'
    FN_TARGET_DOWNSCALE = 'fn_target_downscale'
    DS_TARGET_DOWNSCALE = 'ds_target_downscale'
    MA_TARGET_DOWNSCALE = 'ma_target_downscale'
    FN_CLOUDMASK = 'fn_cloudmask'
    FN_CLOUDMASK_DOWNSCALE = 'fn_cloudmask_downscale'
    DS_CLOUDMASK_DOWNSCALE = 'ds_cloudmask_downscale'
    MA_CLOUDMASK_DOWNSCALE = 'ma_cloudmask_downscale'
    FN_PREFIX = 'fn_prefix'
    FN_COG = 'fn_cog'
    FN_COG_8BAND = 'fn_cog_8band'
    FN_SUFFIX = 'fn_suffix'
    GEOM_TOA = 'geom_toa'

    # File name suffixes
    FN_TOA_SUFFIX = 'fn_toa_suffix'
    FN_TOA_DOWNSCALE_SUFFIX = '_toa_30m.tif'
    FN_TARGET_SUFFIX = 'fn_target_suffix'
    FN_TARGET_DOWNSCALE_SUFFIX = '_target_30m.tif'
    FN_CLOUDMASK_SUFFIX = 'fn_cloudmask_suffix'
    FN_CLOUDMASK_DOWNSCALE_SUFFIX = '_toa_clouds_30m.tif'
    FN_SRLITE_NONCOG_SUFFIX = '_noncog.tif'
    FN_SRLITE_SUFFIX = '_sr_02m.tif'
    FN_WARP_SUFFIX = '_warp.tif'

    # Band pairs
    LIST_BAND_PAIRS = 'list_band_pairs'
    LIST_BAND_PAIR_INDICES = 'list_band_pairs_indices'
    LIST_TOA_BANDS = 'list_toa_bands'
    LIST_TARGET_BANDS = 'list_target_bands'
    BAND_NUM = 'band_num'
    BAND_DESCRIPTION_LIST = 'band_description_list'

    # Index of data arrays FN_LIST, MA_LIST
    LIST_INDEX_TOA = 'list_index_toa'
    LIST_INDEX_TARGET = 'list_index_target'
    LIST_INDEX_CLOUDMASK = 'list_index_cloudmask'
    LIST_INDEX_THRESHOLD = 'list_index_threshold'

    # Target vars and defaults
    TARGET_GEO_TRANSFORM = 'target_geo_transform'
    TARGET_EXTENT = 'target_extent'
    TARGET_FN = 'target_fn'
    TARGET_XRES = 'target_xres'
    TARGET_YRES = 'target_yres'
    TARGET_PRJ = 'target_prj'
    TARGET_SRS = 'target_srs'
    TARGET_RASTERX_SIZE = 'target_rasterX_size'
    TARGET_RASTERY_SIZE = 'target_rasterY_size'
    TARGET_RASTER_COUNT = 'target_raster_count'
    TARGET_DRIVER = 'target_driver'
    TARGET_OUTPUT_TYPE = 'target_output_type'
    TARGET_DTYPE = 'target_dtype'
    TARGET_NODATA_VALUE = 'target_nodata_value'
    TARGET_SAMPLING_METHOD = 'target_sampling_method'

    # Default values
    DEFAULT_TOA_SUFFIX = 'toa.tif'
    DEFAULT_TARGET_SUFFIX = 'ccdc.tif'
    DEFAULT_CLOUDMASK_SUFFIX = 'toa.cloudmask.v1.2.tif'

    # Suffixs modified as per PM - 05/19/24
    DEFAULT_ERROR_REPORT_SUFFIX = 'sr_errors.csv'
    DEFAULT_STATISTICS_REPORT_SUFFIX = '_sr_stats.csv'
    DEFAULT_XRES = 30
    DEFAULT_YRES = 30
    DEFAULT_NODATA_VALUE = -9999
    DEFAULT_SAMPLING_METHOD = 'average'

    # Regression algorithms
    REGRESSION_MODEL = 'regressor'
    REGRESSOR_MODEL_OLS = 'ols'
    REGRESSOR_MODEL_HUBER = 'huber'
    REGRESSOR_MODEL_RMA = 'rma'

    # Storage type
    STORAGE_TYPE = 'storage'
    STORAGE_TYPE_MEMORY = 'memory'
    STORAGE_TYPE_FILE = 'file'

    # Debug & log values
    DEBUG_NONE_VALUE = 0
    DEBUG_TRACE_VALUE = 1
    DEBUG_VIZ_VALUE = 2
    DEBUG_LEVEL = 'debug_level'
    LOG_FLAG = 'log_flag'
    CLEAN_FLAG = 'clean_flag'
    NONCOG_FLAG = 'noncog_flag'
    CSV_FLAG = 'csv_flag'
    ERROR_REPORT_FLAG = 'error_report_flag'
    BAND8_FLAG = 'band8_flag'
    CSV_WRITER = 'csv_writer'

    # Quality flag and list of values
    QUALITY_MASK_FLAG = 'qf_mask_flag'
    LIST_QUALITY_MASK = 'list_quality_mask'
    POSITIVE_MASK_FLAG = 'positive_mask_flag'

    # Cloud mask flag
    CLOUD_MASK_FLAG = 'cloud_mask_flag'

    # Threshold flag
    THRESHOLD_MASK_FLAG = 'threshold_mask_flag'
    THRESHOLD_MIN = 'threshold_min'
    THRESHOLD_MAX = 'threshold_max'

    # Global instance variables
    context_dict = {}
    plotLib = None
    debug_level = 0
    writer = None

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self):

        args = self._getParser()
        # Initialize serializable context for orchestration
        try:
            self.context_dict[Context.BATCH_NAME] = str(args.batch_name)
            self.context_dict[Context.DIR_TOA] = str(args.toa_dir)
            self.context_dict[Context.DIR_TARGET] = str(args.target_dir)
            self.context_dict[Context.DIR_CLOUDMASK] = str(args.cloudmask_dir)

            # Manage output paths
            self.context_dict[Context.DIR_OUTPUT] = str(args.out_dir)
            try:
                os.makedirs(self.context_dict[Context.DIR_OUTPUT], exist_ok=True)
            except OSError as error:
                print("Directory '%s' can not be created" % self.context_dict[Context.DIR_OUTPUT])

            if (args.err_dir == "./"):
                self.context_dict[Context.DIR_OUTPUT_ERROR] = self.context_dict[Context.DIR_OUTPUT]
            else:
                self.context_dict[Context.DIR_OUTPUT_ERROR] = str(args.err_dir)
                try:
                    os.makedirs(self.context_dict[Context.DIR_OUTPUT_ERROR], exist_ok=True)
                except OSError as error:
                    print("Directory '%s' can not be created" % self.context_dict[Context.DIR_OUTPUT_ERROR])

            if (args.warp_dir == "./"):
                self.context_dict[Context.DIR_OUTPUT_WARP] = self.context_dict[Context.DIR_OUTPUT]
            else:
                self.context_dict[Context.DIR_OUTPUT_WARP] = str(args.warp_dir)
                try:
                    os.makedirs(self.context_dict[Context.DIR_OUTPUT_WARP], exist_ok=True)
                except OSError as error:
                    print("Directory '%s' can not be created" % self.context_dict[Context.DIR_OUTPUT_WARP])

            if (args.csv_dir == "./"):
                self.context_dict[Context.DIR_OUTPUT_CSV] = self.context_dict[Context.DIR_OUTPUT]
            else:
                self.context_dict[Context.DIR_OUTPUT_CSV] = str(args.csv_dir)
                try:
                    os.makedirs(self.context_dict[Context.DIR_OUTPUT_CSV], exist_ok=True)
                except OSError as error:
                    print("Directory '%s' can not be created" % self.context_dict[Context.DIR_OUTPUT_CSV])

            # Parse general configuration parameters
            self.context_dict[Context.LIST_BAND_PAIRS] = str(args.band_pairs_list)
            self.context_dict[Context.TARGET_XRES] = int(args.target_xres)
            self.context_dict[Context.TARGET_YRES] = int(args.target_yres)
            self.context_dict[Context.TARGET_SAMPLING_METHOD] = str(args.target_sampling_method)

            self.context_dict[Context.FN_TOA_SUFFIX] = '-' + str(args.toa_suffix)
            self.context_dict[Context.FN_TARGET_SUFFIX] = '-' + str(args.target_suffix)
            self.context_dict[Context.FN_CLOUDMASK_SUFFIX] = '-' + str(args.cloudmask_suffix)

            self.context_dict[Context.REGRESSION_MODEL] = str(args.regressor)
            self.context_dict[Context.DEBUG_LEVEL] = int(args.debug_level)
            self.context_dict[Context.CLEAN_FLAG] = str(args.cleanbool)
            self.context_dict[Context.NONCOG_FLAG] = str(args.noncogbool)
            self.context_dict[Context.LOG_FLAG] = str(args.logbool)
            if eval(self.context_dict[Context.LOG_FLAG]):
                self._create_logfile(self.context_dict[Context.REGRESSION_MODEL],
                                     self.context_dict[Context.DIR_OUTPUT])
            if (int(self.context_dict[Context.DEBUG_LEVEL]) >= int(self.DEBUG_TRACE_VALUE)):
                print(sys.path)
            # self.context_dict[Context.ALGORITHM_CLASS] = str(args.algorithm)
            self.context_dict[Context.STORAGE_TYPE] = str(args.storage)
            self.context_dict[Context.CLOUD_MASK_FLAG] = str(args.cmaskbool)
            self.context_dict[Context.POSITIVE_MASK_FLAG] = str(args.pmaskbool)
            self.context_dict[Context.CSV_FLAG] = str(args.errorreportbool)
            self.context_dict[Context.ERROR_REPORT_FLAG] = str(args.csvbool)
            self.context_dict[Context.BAND8_FLAG] = str(args.band8bool)
            self.context_dict[Context.QUALITY_MASK_FLAG] = str(args.qfmaskbool)
            self.context_dict[Context.LIST_QUALITY_MASK] = str(args.qfmask_list)

            self.context_dict[Context.THRESHOLD_MASK_FLAG] = str(args.thmaskbool)
            threshold_range = (str(args.threshold_range)).partition(",")
            self.context_dict[Context.THRESHOLD_MIN] = int(threshold_range[0])
            self.context_dict[Context.THRESHOLD_MAX] = int(threshold_range[2])

        except BaseException as err:
            print('Check arguments: ', err)
            sys.exit(1)

        # Initialize instance variables
        self.debug_level = int(self.context_dict[Context.DEBUG_LEVEL])
        plotLib = self.plot_lib = PlotLib(self.context_dict[Context.DEBUG_LEVEL])

        # Echo input parameter values
        plotLib.trace(f'Initializing SRLite Regression script with the following parameters')
        plotLib.trace(f'Batch:    {self.context_dict[Context.BATCH_NAME]}')
        plotLib.trace(f'TOA Directory:    {self.context_dict[Context.DIR_TOA]}')
        plotLib.trace(f'TARGET Directory:    {self.context_dict[Context.DIR_TARGET]}')
        plotLib.trace(f'Cloudmask Directory:    {self.context_dict[Context.DIR_CLOUDMASK]}')
        plotLib.trace(f'Output Directory: {self.context_dict[Context.DIR_OUTPUT]}')
        plotLib.trace(f'Band Pairs:    {self.context_dict[Context.LIST_BAND_PAIRS]}')
        plotLib.trace(f'Regression Model:    {self.context_dict[Context.REGRESSION_MODEL]}')
        plotLib.trace(f'Debug Level: {self.context_dict[Context.DEBUG_LEVEL]}')
        plotLib.trace(f'Clean Flag: {self.context_dict[Context.CLEAN_FLAG]}')
        plotLib.trace(f'NonCog Flag: {self.context_dict[Context.NONCOG_FLAG]}')
        plotLib.trace(f'CSV Flag: {self.context_dict[Context.CSV_FLAG]}')
        plotLib.trace(f'Error Report Flag: {self.context_dict[Context.ERROR_REPORT_FLAG]}')
        plotLib.trace(f'Band8 Flag: {self.context_dict[Context.BAND8_FLAG]}')
        plotLib.trace(f'Log: {self.context_dict[Context.LOG_FLAG]}')
        #       plotLib.trace(f'Storage:    {self.context_dict[Context.STORAGE_TYPE]}')
        if (eval(self.context_dict[Context.CLOUD_MASK_FLAG])):
            plotLib.trace(f'Cloud Mask:    {self.context_dict[Context.CLOUD_MASK_FLAG]}')
        if (eval(self.context_dict[Context.POSITIVE_MASK_FLAG])):
            plotLib.trace(f'Positive Pixels Only Flag:    {self.context_dict[Context.POSITIVE_MASK_FLAG]}')

        if (eval(self.context_dict[Context.QUALITY_MASK_FLAG])):
            plotLib.trace(f'Quality Mask:    {self.context_dict[Context.QUALITY_MASK_FLAG]}')
            plotLib.trace(f'Quality Mask Values:    {self.context_dict[Context.LIST_QUALITY_MASK]}')
        if (eval(self.context_dict[Context.THRESHOLD_MASK_FLAG])):
            plotLib.trace(f'Threshold Mask:    {self.context_dict[Context.THRESHOLD_MASK_FLAG]}')
            plotLib.trace(f'Threshold Min:    {self.context_dict[Context.THRESHOLD_MIN]}')
            plotLib.trace(f'Threshold Max:    {self.context_dict[Context.THRESHOLD_MAX]}')
        
        plotLib.trace(f'Output Directory: {self.context_dict[Context.DIR_OUTPUT]}')
        if (self.context_dict[Context.DIR_OUTPUT_ERROR] != self.context_dict[Context.DIR_OUTPUT]):
            plotLib.trace(f'Error Directory: {self.context_dict[Context.DIR_OUTPUT_ERROR]}')
        if (self.context_dict[Context.DIR_OUTPUT_WARP] != self.context_dict[Context.DIR_OUTPUT]):
            plotLib.trace(f'Interim Directory: {self.context_dict[Context.DIR_OUTPUT_WARP]}')
        if (self.context_dict[Context.DIR_OUTPUT_CSV] != self.context_dict[Context.DIR_OUTPUT]):
            plotLib.trace(f'CSV Directory: {self.context_dict[Context.DIR_OUTPUT_CSV]}')
           
        if (eval(self.context_dict[Context.CLEAN_FLAG])):
            path = os.path.join(self.context_dict[Context.DIR_OUTPUT_ERROR],
                                    Context.DEFAULT_ERROR_REPORT_SUFFIX)
            if os.path.exists(path):
                os.remove(path)

            #TODO 
            # path = os.path.join(self.context_dict[Context.DIR_OUTPUT_WARP],
            #                         Context.DEFAULT_ERROR_REPORT_SUFFIX)
            # if os.path.exists(path):
            #     os.remove(path)

            path = os.path.join(self.context_dict[Context.DIR_OUTPUT_CSV],
                                    Context.DEFAULT_STATISTICS_REPORT_SUFFIX)
            if os.path.exists(path):
                os.remove(path)
 

        return

    # -------------------------------------------------------------------------
    # getParser()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def _getParser(self):
        """
        :return: argparser object with CLI commands.
        """
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--batch", "--batch", type=str, required=False, dest='batch_name',
            default=None, help="Specify batch name for run."
        )
        parser.add_argument(
            "-toa_dir", "--input-toa-dir", type=str, required=True, dest='toa_dir',
            default=None, help="Specify directory path containing TOA files."
        )
        parser.add_argument(
            "-target_dir", "--input-target-dir", type=str, required=False, dest='target_dir',
            default=None, help="Specify directory path containing TARGET files."
        )
        parser.add_argument(
            "-cloudmask_dir", "--input-cloudmask-dir", type=str, required=False, dest='cloudmask_dir',
            default=None, help="Specify directory path containing Cloudmask files."
        )
        parser.add_argument(
            "-bandpairs", "--input-list-of-band-pairs", type=str, required=False, dest='band_pairs_list',
            default="[['blue_target', 'BAND-B'], ['green_target', 'BAND-G'], ['red_target', 'BAND-R'], ['nir_target', 'BAND-N']]",
            help="Specify list of band pairs to be processed per scene."
        )
        parser.add_argument(
            "-output_dir", "--output-directory", type=str, required=False, dest='out_dir',
            default="./", help="Specify output directory."
        )
        parser.add_argument(
            "--err_dir", "--output-err-dir", type=str, required=False, dest='err_dir',
            default="./", help="Specify directory path containing error files (defaults to out_dir)."
        )
        parser.add_argument(
            "--warp_dir", "--interim-warp-dir", type=str, required=False, dest='warp_dir',
            default="./", help="Specify directory path containing interim warped files (defaults to out_dir)."
        )
        parser.add_argument(
            "--csv_dir", "--output-csv-dir", type=str, required=False, dest='csv_dir',
            default="./", help="Specify directory path containing statistics files (defaults to out_dir)."
        )
        parser.add_argument(
            "--xres", "--input-x-resolution", type=str, required=False, dest='target_xres',
            default=Context.DEFAULT_XRES, help="Specify target X resolution (default = 30)."
        )
        parser.add_argument(
            "--yres", "--input-y-resolution", type=str, required=False, dest='target_yres',
            default=Context.DEFAULT_XRES, help="Specify target Y resolution (default = 30)."
        )
        parser.add_argument(
            "--sampling", "--reprojection-sampling-method", type=str, required=False, dest='target_sampling_method',
            default=Context.DEFAULT_SAMPLING_METHOD, help="Specify target warp sampling method (default = 'average'')."
        )
        parser.add_argument(
            "--toa_suffix", "--input-toa-suffix", type=str, required=False, dest='toa_suffix',
            default=Context.DEFAULT_TOA_SUFFIX, help="Specify TOA file suffix (default = -toa.tif')."
        )
        parser.add_argument(
            "--target_suffix", "--input-target-suffix", type=str, required=False, dest='target_suffix',
            default=Context.DEFAULT_TARGET_SUFFIX, help="Specify TARGET file suffix (default = -ccdc.tif')."
        )
        parser.add_argument(
            "--cloudmask_suffix", "--input-cloudmask-suffix", type=str, required=False, dest='cloudmask_suffix',
            default=Context.DEFAULT_CLOUDMASK_SUFFIX,
            help="Specify CLOUDMASK file suffix (default = -toa.cloudmask.v1.2.tif')."
        )
        parser.add_argument(
            "--debug", "--debug_level", type=int, required=False, dest='debug_level',
            default=Context.DEBUG_NONE_VALUE, help="Specify debug level [0,1,2,3]"
        )
        parser.add_argument(
            "--clean", "--clean", required=False, dest='cleanbool',
            action='store_true', help="Force cleaning of generated artifacts prior to run (e.g, warp files)."
        )
        #NOTE:  As per MC (3/19/24) regarding noncog flag: "Given your previous testing results that showed no real down side to COG,
        # I don’t see any reason to give the user the choice.  I would rather not offer options that may be confusing unless the 
        # users start to request them."  Functionality exists but should not be advertised to users in documentation.
        parser.add_argument(
            "--noncog", "--noncog", required=False, dest='noncogbool',
            action='store_true', help="Disable Cloud-optimized Geotiff format."
        )
        parser.add_argument(
            "--log", "--log", required=False, dest='logbool',
            action='store_true', help="Set logging."
        )
        parser.add_argument('--regressor',
                            required=False,
                            dest='regressor',
                            default='robust',
                            choices=['ols', 'huber', 'rma'],
                            help='Choose which regression algorithm to use')

        parser.add_argument('--pmask',
                            required=False,
                            dest='pmaskbool',
                            default=False,
                            action='store_true',
                            help='Suppress negative pixel values in reprojected bands')

        parser.add_argument('--csv',
                            required=False,
                            dest='csvbool',
                            default=True,
                            action='store_true',
                            help='Generate CSV file with runtime history')

        parser.add_argument('--err',
                            required=False,
                            dest='errorreportbool',
                            default=True,
                            action='store_true',
                            help='Generate error report')

        parser.add_argument('--band8',
                            required=False,
                            dest='band8bool',
                            default=False,
                            action='store_true',
                            help='Generate missing spectral bands [Coastal,Yellow,Rededge,NIR2] ' \
                                 + 'when using CCDC as the target  - use only when input = [Blue|Green|Red|NIR]')

        parser.add_argument('--storage',
                            required=False,
                            dest='storage',
                            default='memory',
                            choices=['memory', 'file'],
                            help='Choose which storage model to use')

        parser.add_argument('--cloudmask',
                            required=False,
                            dest='cmaskbool',
                            default=False,
                            action='store_true',
                            help='Apply cloud mask values to common mask')

        parser.add_argument('--qfmask',
                            required=False,
                            dest='qfmaskbool',
                            default=False,
                            action='store_true',
                            help='Apply quality flag values to common mask')

        parser.add_argument('--qfmasklist',
                            required=False,
                            dest='qfmask_list',
                            default='0,3,4',
                            type=str,
                            help='Choose quality flag values to mask')

        parser.add_argument('--thmask',
                            required=False,
                            dest='thmaskbool',
                            default=False,
                            action='store_true',
                            help='Apply threshold mask values to common mask')

        parser.add_argument('--thrange',
                            required=False,
                            dest='threshold_range',
                            default='-100, 2000',
                            type=str,
                            help='Choose quality flag values to mask')

        return parser.parse_args()

    # -------------------------------------------------------------------------
    # getDict()
    #
    # Get context dictionary
    # -------------------------------------------------------------------------
    def getDict(self):
        return self.context_dict

    # -------------------------------------------------------------------------
    # getPlotLib()
    #
    # Get handle to plotting library
    # -------------------------------------------------------------------------
    def getPlotLib(self):
        return self.plot_lib

    # # -------------------------------------------------------------------------
    # # getDebugLevel()
    # #
    # # Get debug_level
    # # -------------------------------------------------------------------------
    # def getDebugLevel(self):
    #     return self.debug_level

    # -------------------------------------------------------------------------
    # getFileNames()
    #
    # Get input file names
    # -------------------------------------------------------------------------
    def getFileNames(self, prefix, context):
        """
        :param prefix: core TOA file name (must match core target and cloudmask file name)
        :param context: input context object dictionary
        :return: updated context
        """
        context[Context.FN_PREFIX] = str((prefix[1]).split("-toa.tif", 1)[0])
        last_index = context[Context.FN_PREFIX].rindex('_')
        context[Context.CAT_ID] =  context[Context.FN_PREFIX] [last_index+1:]

        # Provide the fully-qualified file name (if provided).  Otherwise assume, list of files
        if os.path.isfile(Path(context[Context.DIR_TOA])):
            context[Context.FN_TOA] = context[Context.DIR_TOA]
        else:
            context[Context.FN_TOA] = os.path.join(context[Context.DIR_TOA] + '/' +
                                                   context[Context.FN_PREFIX] + context[Context.FN_TOA_SUFFIX])

        if os.path.isfile(Path(context[Context.DIR_TARGET])):
            context[Context.FN_TARGET] = context[Context.DIR_TARGET]
        else:
            context[Context.FN_TARGET] = os.path.join(context[Context.DIR_TARGET] + '/' +
                                                      context[Context.FN_PREFIX] + context[Context.FN_TARGET_SUFFIX])

        if os.path.isfile(Path(context[Context.DIR_CLOUDMASK])):
            context[Context.FN_CLOUDMASK] = context[Context.DIR_CLOUDMASK]
        else:
            context[Context.FN_CLOUDMASK] = os.path.join(context[Context.DIR_CLOUDMASK] + '/' +
                                                         context[Context.FN_PREFIX] + context[
                                                             Context.FN_CLOUDMASK_SUFFIX])

        # Name artifacts according to TOA prefix
        context[Context.FN_TOA_DOWNSCALE] = os.path.join(context[Context.DIR_OUTPUT] + '/' +
                                                         context[Context.FN_PREFIX] + self.FN_TOA_DOWNSCALE_SUFFIX)
        context[Context.FN_TARGET_DOWNSCALE] = os.path.join(context[Context.DIR_OUTPUT] + '/' +
                                                            context[
                                                                Context.FN_PREFIX] + self.FN_TARGET_DOWNSCALE_SUFFIX)
        context[Context.FN_CLOUDMASK_DOWNSCALE] = os.path.join(context[Context.DIR_OUTPUT] + '/' +
                                                               context[
                                                                   Context.FN_PREFIX] + self.FN_CLOUDMASK_DOWNSCALE_SUFFIX)

                                                    
        if (eval(self.context_dict[Context.NONCOG_FLAG])):
            context[Context.FN_COG] = os.path.join(context[Context.DIR_OUTPUT] + '/' +
                                               context[Context.FN_PREFIX] + self.FN_SRLITE_NONCOG_SUFFIX)
        else:
            context[Context.FN_COG] = os.path.join(context[Context.DIR_OUTPUT] + '/' +
                                               context[Context.FN_PREFIX] + self.FN_SRLITE_SUFFIX)


        if not (os.path.exists(context[Context.FN_TOA])):
            raise FileNotFoundError("TOA File not found: {}".format(context[Context.FN_TOA]))
        if not (os.path.exists(context[Context.FN_TARGET])):
            self.plot_lib.trace("Processing: " + context[Context.FN_TOA])
            raise FileNotFoundError("TARGET File not found: {}".format(context[Context.FN_TARGET]))
        if (eval(self.context_dict[Context.CLOUD_MASK_FLAG])):
            if not (os.path.exists(context[Context.FN_CLOUDMASK])):
                self.plot_lib.trace("Processing: " + context[Context.FN_TOA])
                raise FileNotFoundError("Cloudmask File not found: {}".format(context[Context.FN_CLOUDMASK]))

        return context

    # -------------------------------------------------------------------------
    # _create_logfile()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def _create_logfile(self, model, logdir='results'):
        """
        :param args: argparser object
        :param logdir: log directory to store log file
        :return: logfile instance, stdour and stderr being logged to file
        """
        logfile = os.path.join(logdir, '{}_log_{}_model.txt'.format(
            datetime.now().strftime("%Y%m%d-%H%M%S"), model))
        print('See ', logfile)
        so = se = open(logfile, 'w')  # open our log file
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # stdout buffering
        os.dup2(so.fileno(), sys.stdout.fileno())  # redirect to the log file
        os.dup2(se.fileno(), sys.stderr.fileno())
        return logfile
