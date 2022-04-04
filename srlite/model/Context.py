#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse  # system libraries
from datetime import datetime
from srlite.model.PlotLib import PlotLib

# -----------------------------------------------------------------------------
# class Context
#
# This class is a serializable context for orchestration.
# -----------------------------------------------------------------------------
class Context(object):

    # Directories
    DIR_TOA = 'dir_toa'
    DIR_CCDC = 'dir_ccdc'
    DIR_CLOUDMASK = 'dir_cloudmask'
    DIR_WARP = 'dir_warp'
    DIR_OUTPUT = 'dir_out'

    # File names
    FN_DEST = 'fn_dest'
    FN_SRC = 'fn_src'

    FN_TOA = 'fn_toa'
    FN_CCDC = 'fn_ccdc'
    FN_CLOUDMASK = 'fn_cloudmask'
    FN_WARP = 'fn_warp'
    FN_PREFIX = 'fn_prefix'

    # File name suffixes
    FN_TOA_SUFFIX = '-toa.tif'
    FN_CCDC_SUFFIX = '-ccdc.tif'
    FN_CLOUDMASK_SUFFIX = '-toa_pred.tif'
    FN_CLOUDMASK_WARP_SUFFIX = '-toa_pred_warp.tif'

    # Band pairs
    LIST_BAND_PAIRS = 'band_pairs_list'

    # Target vars and defaults
    TARGET_ATTR = 'target_attr'
    TARGET_EXTENT = 'target_extent'
    TARGET_XRES = 'target_xres'
    TARGET_YRES = 'target_yres'
    TARGET_PRJ = 'target_prj'
    TARGET_SRS = 'target_srs'
    TARGET_OUTPUT_TYPE = 'target_output_type'

    DEFAULT_XRES = 30.0
    DEFAULT_YRES = 30.0

    # Regression algorithms
    REGRESSOR_SIMPLE = 'simple'
    REGRESSOR_ROBUST = 'robust'
    REGRESSION_MODEL = 'regressor'

    # Debug & log values
    DEBUG_NONE_VALUE = 0
    DEBUG_TRACE_VALUE = 1
    DEBUG_VIZ_VALUE = 2
    DEBUG_LEVEL = 'debug_level'
    LOG_FLAG = 'log_flag'
    CLEAN_FLAG = 'clean_log'

    # Global instance variables
    context_dict = {}
    plotLib = None
    debug_level = 0

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self):

        args = self._getParser()
        # Initialize serializable context for orchestration
        try:
            self.context_dict[Context.DIR_TOA] = str(args.toa_dir)
            self.context_dict[Context.DIR_CCDC] = str(args.ccdc_dir)
            self.context_dict[Context.DIR_CLOUDMASK] = str(args.cloudmask_dir)
            self.context_dict[Context.DIR_OUTPUT] = str(args.out_dir)
            self.context_dict[Context.DIR_WARP] = self.context_dict[Context.DIR_OUTPUT]
            if not (args.warp_dir == None):
                self.context_dict[Context.DIR_WARP] = str(args.warp_dir)

            self.context_dict[Context.LIST_BAND_PAIRS] = str(args.band_pairs_list)
            self.context_dict[Context.TARGET_XRES] = self.DEFAULT_XRES
            if not (args.target_xres == None):
                self.context_dict[Context.TARGET_YRES] = float(args.target_xres)
            self.context_dict[Context.TARGET_YRES] = self.DEFAULT_YRES
            if not (args.target_yres == None):
                self.context_dict[Context.TARGET_YRES] = float(args.target_yres)
            self.context_dict[Context.REGRESSION_MODEL] = str(args.regressor)
            self.context_dict[Context.DEBUG_LEVEL] = int(args.debug_level)
            self.context_dict[Context.CLEAN_FLAG] = str(args.cleanbool)
            self.context_dict[Context.LOG_FLAG] = str(args.logbool)
            if eval(self.context_dict[Context.LOG_FLAG]):
                self._create_logfile(self.context_dict[Context.REGRESSION_MODEL],
                                     self.context_dict[Context.DIR_OUTPUT])
            if (int(self.context_dict[Context.DEBUG_LEVEL]) >= int(self.DEBUG_TRACE_VALUE)):
                    print(sys.path)

        except BaseException as err:
            print('Check arguments: ', err)
            sys.exit(1)

        # Initialize instance variables
        self.debug_level = int(self.context_dict[Context.DEBUG_LEVEL])
        plotLib = self.plot_lib = PlotLib(self.context_dict[Context.DEBUG_LEVEL])
        os.system(f'mkdir -p {self.context_dict[Context.DIR_OUTPUT]}')
        os.system(f'mkdir -p {self.context_dict[Context.DIR_WARP]}')

        # Echo input parameter values
        plotLib.trace(f'Initializing SRLite Regression script with the following parameters')
        plotLib.trace(f'TOA Directory:    {self.context_dict[Context.DIR_TOA]}')
        plotLib.trace(f'CCDC Directory:    {self.context_dict[Context.DIR_CCDC]}')
        plotLib.trace(f'Cloudmask Directory:    {self.context_dict[Context.DIR_CLOUDMASK]}')
        plotLib.trace(f'Warp Directory:    {self.context_dict[Context.DIR_WARP]}')
        plotLib.trace(f'Output Directory: {self.context_dict[Context.DIR_OUTPUT]}')
        plotLib.trace(f'Band Pairs:    {self.context_dict[Context.LIST_BAND_PAIRS]}')
        plotLib.trace(f'Regression Model:    {self.context_dict[Context.REGRESSION_MODEL]}')
        plotLib.trace(f'Debug Level: {self.context_dict[Context.DEBUG_LEVEL]}')
        plotLib.trace(f'Clean Flag: {self.context_dict[Context.CLEAN_FLAG]}')
        plotLib.trace(f'Log: {self.context_dict[Context.LOG_FLAG]}')

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
            "-toa", "--input-toa-dir", type=str, required=True, dest='toa_dir',
            default=None, help="Specify directory path containing TOA files."
        )
        parser.add_argument(
            "-ccdc", "--input-ccdc-dir", type=str, required=False, dest='ccdc_dir',
            default=None, help="Specify directory path containing CCDC files."
        )
        parser.add_argument(
            "-cloudmask", "--input-cloudmask-dir", type=str, required=True, dest='cloudmask_dir',
            default=None, help="Specify directory path containing Cloudmask files."
        )
        parser.add_argument(
            "-bandpairs", "--input-list-of-band-pairs", type=str, required=False, dest='band_pairs_list',
            default="[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N']]",
            help="Specify list of band pairs to be processed per scene."
        )
        parser.add_argument(
            "-o", "--output-directory", type=str, required=False, dest='out_dir',
            default="./", help="Specify output directory."
        )
        parser.add_argument(
            "--warp", "--input-warp-dir", type=str, required=False, dest='warp_dir',
            default=None, help="Specify directory path containing wapred files."
        )
        parser.add_argument(
            "--xres", "--input-x-resolution", type=str, required=False, dest='target_xres',
            default=None, help="Specify target X resolution (default = 30.0)."
        )
        parser.add_argument(
            "--yres", "--input-y-resolution", type=str, required=False, dest='target_yres',
            default=None, help="Specify target Y resolution (default = 30.0)."
        )
        parser.add_argument(
            "--debug", "--debug_level", type=int, required=False, dest='debug_level',
            default=0, help="Specify debug level [0,1,2,3]"
        )
        parser.add_argument(
            "--fc", "--fc", required=False, dest='cleanbool',
            action='store_true', help="Force cleaning of generated artifacts prior to run (e.g, warp files)."
        )
        parser.add_argument(
            "--log", "--log", required=False, dest='logbool',
            action='store_true', help="Set logging."
        )
        parser.add_argument('--regressor',
                            required=False,
                            dest='regressor',
                            default='robust',
                            choices=['simple', 'robust'],
                            help='Choose which regression algorithm to use')

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

    # -------------------------------------------------------------------------
    # getDebugLevel()
    #
    # Get debug_level
    # -------------------------------------------------------------------------
    def getDebugLevel(self):
        return self.debug_level

    # -------------------------------------------------------------------------
    # getFileNames()
    #
    # Get input file names
    # -------------------------------------------------------------------------
    def getFileNames(self, prefix, context):
        """
        :param prefix: core TOA file name (must match core ccdc and cloudmask file name)
        :param context: input context object dictionary
        :return: updated context
        """
        context[Context.FN_PREFIX] = str((prefix[1]).split("-toa.tif", 1)[0])
        context[Context.FN_TOA] = os.path.join(context[Context.DIR_TOA] + '/' +
                                               context[Context.FN_PREFIX] + self.FN_TOA_SUFFIX)
        context[Context.FN_CCDC] = os.path.join(context[Context.DIR_CCDC] + '/' +
                                                context[Context.FN_PREFIX] + self.FN_CCDC_SUFFIX)
        context[Context.FN_CLOUDMASK] = os.path.join(context[Context.DIR_CLOUDMASK] + '/' +
                                                     context[Context.FN_PREFIX] + self.FN_CLOUDMASK_SUFFIX)
        context[Context.FN_WARP] = os.path.join(context[Context.DIR_WARP] + '/' +
                                                context[Context.FN_PREFIX] + self.FN_CLOUDMASK_WARP_SUFFIX)

        return context

    # -------------------------------------------------------------------------
    # create_logfile()
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
