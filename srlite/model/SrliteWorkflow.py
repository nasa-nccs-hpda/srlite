import os
import time

from srlite.model.Context import Context
from srlite.model.RasterLib import RasterLib

import pandas as pd
import ast
from pygeotools.lib import iolib
import numpy as np

from pathlib import Path

# -----------------------------------------------------------------------------
# class SrliteWorkflow
#
# processToa(toa)
#
# processToas()
#    processToa(toa)
# -----------------------------------------------------------------------------
class SrliteWorkflow(RasterLib):
    """This class orchestrates Srlite workflows`.

    Required Parameters:
    :param output_dir: Directory to store output files.
    :type output_dir: str
    :param toa_src: Directory that hosts TOA files -OR- a specific TOA (fully qualified path).
    :type toa_src: str
    :param target_dir: Directory that hosts Reference files (e.g., CCDC).  File name generated from TOA prefix.
    :type target_dir: str
    :param cloudmask_dir: Directory that hosts Cloudmask files.  File name generated from TOA prefix.
    :type cloudmask_dir: str

    Optional Parameters:
    :param regressor: Regression algorithm to use (i.e., "ols", "huber", "rma").
    :type regressor: str
    :param debug: Specify debug level [0,1,2,3]
    :type debug: int
    :param pmask: Suppress negative pixel values in reprojected bands. ["True", "False"]
    :type pmask: str
    :param cloudmask: Apply cloud mask values to common mask. ["True", "False"]
    :type cloudmask: str
    :param csv: Directory path containing statistics files (defaults to out_dir).
    :type csv: str
    :param band8: 'Generate missing spectral bands [Coastal,Yellow,Rededge,NIR2] when using CCDC as the Reference  - use only when input = [Blue|Green|Red|NIR]'
    :type band8: str
    :param clean: Force cleaning of generated artifacts prior to run (e.g, warp files). ["True", "False"]
    :type clean: str
    :param cloudmask_suffix=: CLOUDMASK file suffix (default = -toa.cloudmask.v1.2.tif').
    :type cloudmask_suffix=: str
    :param target_suffix=: TARGET file suffix (default = -ccdc.tif').
    :type target_suffix=: str
    :param logger: A logger device
    :type logger: str
    """

    # -------------------------------------------------------------------------
    # __validatePaths__ : Verify user-specified paths
    # -------------------------------------------------------------------------
    def _validatePaths(self):
        err = None

       # Validate paths
        if (self.context[Context.DIR_OUTPUT] != None): 
                try:
                    os.makedirs(self.context[Context.DIR_OUTPUT], exist_ok=True)
                except OSError as error:
                    err = ('\nCannot create output directory: ', self.context[Context.DIR_OUTPUT])
        if (self.context[Context.DIR_TOA] != None): 
            if not os.path.exists(self.context[Context.DIR_TOA]):
                err = ('\nTOA Path or file does not exist: ', self.context[Context.DIR_TOA])             
        if (self.context[Context.DIR_TARGET] != None): 
            if not os.path.exists(self.context[Context.DIR_TARGET]):
                err = ('\nReference Path or file does not exist: ', self.context[Context.DIR_TARGET])             
        if (self.context[Context.DIR_CLOUDMASK] != None): 
             if not os.path.exists(self.context[Context.DIR_CLOUDMASK]):
                err = ('\nCloudmask Path or file does not exist: ', self.context[Context.DIR_CLOUDMASK])

        if (err != None):
             raise Exception(err)             

    # -------------------------------------------------------------------------
    # __init__ : Constructor that initializes Srlite Workflow
    #   _validatePaths()
    # -------------------------------------------------------------------------
    def __init__(self, 
                output_dir=None, 
                toa_src=None, 
                target_dir=None, 
                cloudmask_dir=None, 
                regressor="rma",
                debug=0,
                pmask=True,
                cloudmask=True,
                csv=True,
                band8=True,
                clean=True,
                bandpairs="[['blue_ccdc', 'BAND-B'], ['green_ccdc', 'BAND-G'], ['red_ccdc', 'BAND-R'], ['nir_ccdc', 'BAND-N'],['blue_ccdc', 'BAND-C'], ['green_ccdc', 'BAND-Y'], ['red_ccdc', 'BAND-RE'], ['nir_ccdc', 'BAND-N2']]",
#                bandpairs="[['BLUE', 'BAND-B'], ['GREEN', 'BAND-G'], ['RED', 'BAND-R'], ['NIR', 'BAND-N'],['BLUE', 'BAND-C'], ['GREEN', 'BAND-Y'], ['RED', 'BAND-RE'], ['NIR', 'BAND-N2']]",
                toa_suffix="-toa.tif",
                target_suffix="ccdc.tif",
                cloudmask_suffix="toa.cloudmask.tif", 
                logger=None):
        
        self.start_time = time.time()  # record start time
        self._logger = logger

        # Initialize context
        try:
            self.contextClazz = Context(output_dir=output_dir, overrideArgs=True, debug=debug)
            self.context = self.contextClazz.getDict()
        except BaseException as err:
            print('\nError during context initialization', str(err))
            raise err
 
        # Get handles to plot and raster classes
        self.plot_lib = self.plotLib = self.contextClazz.getPlotLib()
        self.raster_lib = self.rasterLib = RasterLib(int(self.context[Context.DEBUG_LEVEL]), self.plotLib)

        # Retrieve suffixes
        self.context[Context.FN_TOA_SUFFIX] = toa_suffix
        self.context[Context.FN_TARGET_SUFFIX] = target_suffix 
        self.context[Context.FN_CLOUDMASK_SUFFIX] = cloudmask_suffix 

        # # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
        # self.context = self.contextClazz.getFileNames(str(self.context[Context.FN_TOA]).rsplit("/", 1), self.context)

        # Override CLI as needed
        if (output_dir != None): 
                self.context[Context.DIR_OUTPUT] = output_dir
        if (toa_src != None): 
                self.context[Context.DIR_TOA] = toa_src                
        if (target_dir != None): 
                self.context[Context.DIR_TARGET] = target_dir
        if (cloudmask_dir != None): 
                self.context[Context.DIR_CLOUDMASK] = cloudmask_dir
        self._validatePaths()

        self.context[Context.LIST_BAND_PAIRS] = bandpairs
        self.context[Context.REGRESSION_MODEL] = regressor
        self.context[Context.DEBUG_LEVEL] = debug
        self.context[Context.POSITIVE_MASK_FLAG] = pmask
        self.context[Context.CLOUD_MASK_FLAG] = cloudmask
        self.context[Context.ERROR_REPORT_FLAG] = csv
        self.context[Context.BAND8_FLAG] = band8
        self.context[Context.CLEAN_FLAG] = clean



        # Retrieve TOA files in sorted order from the input TOA directory and loop through them
        self.context[Context.LIST_TOAS] = self.contextClazz.getToaList()

        # Echo input parameter values
        self.contextClazz.echoInput(self.plotLib)

        return
    
    # -------------------------------------------------------------------------
    # processToas() - loop through default list of TOAs provided in __init__()
    #    _validatePaths(toa)
    #    processToa(toa)
    # -------------------------------------------------------------------------
    def processToas(self):
        self._validatePaths()
        for toa in self.context[Context.LIST_TOAS]:
             self.processToa(toa)
    
    # -------------------------------------------------------------------------
    # processToa(toa) - process a single TOA (expects a specific, fully-qualified path with file name)
    #
    # Perform regression to capture coefficients from intersected pixels and apply to 2m source
    # -------------------------------------------------------------------------
    def processToa(self, toa):
        errorIndex = 0
        sr_errors_list = []
        rasterLib = self.rasterLib
        plotLib = self.plotLib
        context = self.context
        contextClazz = self.contextClazz
        context[Context.FN_TOA] = toa

        # Verify that TOA exists first, since Reference and Cloudmask file names are derived from TOA file name
        self.context[Context.DIR_TOA] = toa             
        self._validatePaths()

        try:
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(Path(toa)).rsplit("/", 1), context)
            
            # Remove existing SR-Lite output if clean_flag is activated
            rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if not (os.path.exists(context[Context.FN_COG])):

                try:
                    # Capture input attributes - then align all artifacts to EVHR TOA projection
                    rasterLib.getAttributeSnapshot(context)

                    # Define order indices for list processing
                    context[Context.LIST_INDEX_TARGET] = 0
                    context[Context.LIST_INDEX_TOA] = 1
                    context[Context.LIST_INDEX_CLOUDMASK] = -1  # increment if cloudmask requested

                    # Validate that input band name pairs exist in EVHR & CCDC files
                    context[Context.FN_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA])]
                    context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                    #  Reproject (downscale) TOA to CCDC resolution (30m)  - use 'average' for resampling method
                    #  Reproject TARGET (CCDC) to remaining attributes of EVHR TOA Downscale (extent, srs, etc.) 
                    context[Context.FN_REPROJECTION_LIST] = [str(context[Context.FN_TARGET]), str(context[Context.FN_TOA])]
                    context[Context.TARGET_FN] = str(context[Context.FN_TOA])
                    context[Context.TARGET_SAMPLING_METHOD] = 'average'
                    context[Context.DS_WARP_LIST], context[Context.MA_WARP_LIST] = rasterLib.getReprojection(context)

                    #  Reproject cloudmask to attributes of EVHR TOA Downscale  - use 'mode' for resampling method
                    if eval(context[Context.CLOUD_MASK_FLAG]):
                        context[Context.FN_LIST].append(str(context[Context.FN_CLOUDMASK]))
                        context[Context.FN_REPROJECTION_LIST] = [str(context[Context.FN_CLOUDMASK])]
                        context[Context.TARGET_FN] = str(context[Context.FN_TOA])
                            
                        # Reproject to 'mode' sampling for regression
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

                    # Generate CSV
                    if eval(context[Context.CSV_FLAG]):
                        rasterLib.generateCSV(context)

                    # Clean up
                    rasterLib.refresh(context)

                except BaseException as err:
                    print('\nToa processing failed - Error details: ', str(err))
                    ########### save error for each failed TOA #############
                    metadata = {}
                    metadata['toa_name'] = str(context[Context.FN_TOA])
                    metadata['error'] = str(err)
                    if (errorIndex == 0):
                        sr_errors_list = pd.concat([pd.DataFrame([metadata], index=[errorIndex])])
                    else:
                        sr_errors_list = pd.concat([sr_errors_list, pd.DataFrame([metadata], index=[errorIndex])])
                    errorIndex = errorIndex + 1

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
        if eval(str(context[Context.ERROR_REPORT_FLAG])):
            rasterLib.generateErrorReport(context)

        print("\nTotal Elapsed Time for " + str(context[Context.DIR_OUTPUT]) + ': ',
            (time.time() - self.start_time) / 60.0)  # time in min

    def processToaConcurrently(self, toa, contextClazz, context, rasterLib):
            
            errorIndex = 0
            
            # Generate file names based on incoming EVHR file and declared suffixes - get snapshot
            context = contextClazz.getFileNames(str(toa).rsplit("/", 1), context)

            # Remove existing SR-Lite output if clean_flag is activated
            rasterLib.removeFile(context[Context.FN_COG], context[Context.CLEAN_FLAG])

            # Proceed if SR-Lite output does not exist
            if not (os.path.exists(context[Context.FN_COG])):

                try:
                    # Capture input attributes - then align all artifacts to EVHR TOA projection
                    rasterLib.getAttributeSnapshot(context)

                    # Define order indices for list processing
                    context[Context.LIST_INDEX_TARGET] = 0
                    context[Context.LIST_INDEX_TOA] = 1
                    context[Context.LIST_INDEX_CLOUDMASK] = -1  # increment if cloudmask requested

                    # Validate that input band name pairs exist in EVHR & CCDC files
                    context[Context.FN_LIST] = [str(context[Context.FN_TARGET]), str(toa)]
                    context[Context.LIST_BAND_PAIR_INDICES] = rasterLib.getBandIndices(context)

                    # Align inputs to TOA
                    rasterLib.alignInputs(toa, context)

                    # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
                        # sr_prediction_list, context[Context.METRICS_LIST], common_mask_list
                    context[Context.PRED_LIST], context[Context.METRICS_LIST], context[Context.COMMON_MASK_LIST] = \
                        self.simulateSurfaceReflectanceConcurrent(context)

                    # Create COG image from stack of processed bands
                    context[Context.FN_SRC] = str(toa)
                    context[Context.FN_DEST] = str(context[Context.FN_COG])
                    context[Context.BAND_NUM] = len(list(context[Context.LIST_TOA_BANDS]))
                    context[Context.BAND_DESCRIPTION_LIST] = list(context[Context.LIST_TOA_BANDS])
                    context[Context.FN_COG] = rasterLib.createImage(context)

                    # Generate CSV
                    if eval(context[Context.CSV_FLAG]):
                        rasterLib.generateCSV(context)

                    # Clean up
                    rasterLib.refresh(context)
                except BaseException as err:
                    print('\nToa processing failed - Error details: ', err)
                    ########### save error for each failed TOA #############
                    metadata = {}
                    metadata['toa_name'] = str(toa)
                    metadata['error'] = str(err)
                    if (errorIndex == 0):
                        sr_errors_list = pd.concat([pd.DataFrame([metadata], index=[errorIndex])])
                    else:
                        sr_errors_list = pd.concat([sr_errors_list, pd.DataFrame([metadata], index=[errorIndex])])
                    errorIndex = errorIndex + 1
                    
            return
    # -------------------------------------------------------------------------
    # processBandPairIndex()
    #
    # Populate dictionary of coefficients
    # -------------------------------------------------------------------------
    def _processBandPairIndexConcurrent(self, bandPairIndicesList, bandPairIndex, targetBandMaArray, toaBandMaArray, 
                                       bandNamePairList, common_mask_list, minWarning, sr_unmasked_prediction_list, sr_prediction_list, sr_metrics_list):
    # def processBandPairIndexConcurrent(self, bandPairIndicesList, bandPairIndex, warp_ds_list, 
    #                    bandNamePairList, common_mask_list, minWarning, sr_unmasked_prediction_list, sr_prediction_list, sr_metrics_list):
    # def processBandPairIndexConcurrent(self, context):
        # bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
        # bandNamePairList = context

        self._plot_lib.trace('=>')
        self._plot_lib.trace('====================================================================================')
        self._plot_lib.trace('=========== processBandPairIndexConcurrent====')
        self._plot_lib.trace('====================================================================================')

    # -------------------------------------------------------------------------
    # processBandPairIndex()
    #
    # Populate dictionary of coefficients
    # -------------------------------------------------------------------------
    def processBandPairIndexConcurrent(self, reg_model, bandPairIndicesList, bandPairIndex,  
                       bandNamePairList, minWarning, warp_ma_masked_band_list, toaBandMaArrayRaw,
                       sr_unmasked_prediction_list, sr_prediction_list, sr_metrics_list):
            
        try:
            self._plot_lib.trace('=>')
            self._plot_lib.trace('====================================================================================')
            self._plot_lib.trace('== Start Processing Band #' + str(bandPairIndex + 1) + ' ' + 
                                 str(bandPairIndicesList[bandPairIndex + 1]) + ' ===============')
            self._plot_lib.trace('====================================================================================')

            # Retrieve band pair
            bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

            # Check the mins of each ma - they should be greater than 0
            for j, ma in enumerate(warp_ma_masked_band_list):
                j = j + 1
                if (ma.min() < minWarning):
                    self._plot_lib.trace("Warning: Masked array values should be larger than " + str(minWarning))
            #                    exit(1)

            ########################################
            # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
            # CCDC SR is first element in list, which needs to be the y-var:
            # b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
            ########################################

            # Get 2m TOA Masked Array
            # toaIndexArray = bandPairIndicesList[bandPairIndex+1]
            # toaIndex = toaIndexArray[1]
            # toaBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA], toaIndex)
            sr_prediction_band, metadata = self.predictSurfaceReflectanceConcurrent(reg_model,
                                                                          bandNamePairList[bandPairIndex][1],
                                                                          toaBandMaArrayRaw,
                                                                          warp_ma_masked_band_list[0],
                                                                          warp_ma_masked_band_list[1],
                                                                          sr_metrics_list)

            ########################################
            # #### Apply the model to the original EVHR (2m) to predict surface reflectance
            ########################################
            self._plot_lib.trace(
                f'Applying model to {str(bandNamePairList[bandPairIndex])} in file ')
                # f'{os.path.basename(context[Context.FN_LIST][context[Context.LIST_INDEX_TOA]])}')
            self._plot_lib.trace(f'Metrics: {metadata}')

            ########### save predictions for each band #############
            sr_unmasked_prediction_list.append(sr_prediction_band)

            # Return to original shape and apply original mask
            toa_sr_ma_band_reshaped = sr_prediction_band.reshape(toaBandMaArrayRaw.shape)

            toa_sr_ma_band = np.ma.array(
                toa_sr_ma_band_reshaped,
                mask=toaBandMaArrayRaw.mask)
            sr_prediction_list.append(toa_sr_ma_band)

            print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

        except BaseException as err:
                print('\nprocessBandPairIndexConcurrent processing failed - Error details: ', err)
                raise err
    #          band-meta      band-data-list of 1             toa-masked list of 1
        return [metadata, sr_prediction_list]

    # -------------------------------------------------------------------------
    # simulateSurfaceReflectance()
    #
    # Perform workflow to create simulated surface reflectance for each band (SR-Lite)
    # This method hosts the primary orchestration logic for the SR-Lite application.
    # -------------------------------------------------------------------------
    def simulateSurfaceReflectanceConcurrent(self, context):
        self._validateParms(context,
                            [Context.MA_WARP_LIST, Context.LIST_BAND_PAIRS, Context.LIST_BAND_PAIR_INDICES,
                             Context.REGRESSION_MODEL, Context.FN_LIST])

        try: 
            from pathos.multiprocessing import ProcessingPool,ThreadingPool
            # tmap = ThreadingPool().map
            amap = ProcessingPool().amap            
            
            bandPairIndicesList = context[Context.LIST_BAND_PAIR_INDICES]

            sr_prediction_list = []
            sr_unmasked_prediction_list = []
            band_pair_results = []
            # common_mask_list = []

            sr_metrics_list = context[Context.METRICS_LIST] = []
            context[Context.PRED_LIST] = []
            context[Context.COMMON_MASK_LIST] = []
            warp_ds_list = context[Context.DS_WARP_LIST]
            bandNamePairList = list(ast.literal_eval(context[Context.LIST_BAND_PAIRS]))
            minWarning = 0
            reg_model = context[Context.REGRESSION_MODEL] 

            # Aggregate the requested masks (e.g., clouds, quality mask)
            self.prepareMasks(context)

            ########################################
            # ### FOR EACH BAND PAIR,
            # now, each input should have same exact dimensions, grid, projection.
            # They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
            ########################################
            #for bandPairIndex in range(0, len(bandPairIndicesList) - 1):
            num_workers = len(bandPairIndicesList) - 1

            results = []
            for bandPairIndex in range(num_workers):
                print(f'Starting ProcessingPool().amap() for toa: {str(bandPairIndicesList[bandPairIndex+1])}', flush=True)
 
 #?????? Need  next block?
                # Get 30m EVHR/CCDC Masked Arrays
                bandPairIndices = bandPairIndicesList[bandPairIndex + 1]
                targetBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
                toaBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

                # Create common mask based on user-specified list (e.g., cloudmask, threshold, QF)
                context[Context.COMMON_MASK] = self.getCommonMask(context, targetBandMaArray, toaBandMaArray)
                context[Context.COMMON_MASK_LIST].append(context[Context.COMMON_MASK])

                # Apply the 3-way common mask to the CCDC and EVHR bands
                warp_ma_masked_band_list = [np.ma.array(targetBandMaArray, mask=context[Context.COMMON_MASK]),
                                            np.ma.array(toaBandMaArray, mask=context[Context.COMMON_MASK])]

                toaIndexArray = bandPairIndicesList[bandPairIndex+1]
                toaIndex = toaIndexArray[1]
                toaBandMaArrayRaw = iolib.fn_getma(context[Context.FN_TOA], toaIndex)

                result = amap(self.processBandPairIndexConcurrent, [reg_model], [bandPairIndicesList], [bandPairIndex],
                        [bandNamePairList], [minWarning], [warp_ma_masked_band_list], [toaBandMaArrayRaw],
                        [sr_unmasked_prediction_list], [sr_prediction_list], [sr_metrics_list])
                results.append(result)

            for j in range(num_workers):
                band_pair_results.append(results[j].get())

            for k in range(num_workers):
                temp = band_pair_results[k]
                _metadata = temp[0][0]
                _sr_prediction_list = temp[0][1]

                ########### CRITICAL SECTION - save metadata for each band #############
                context[Context.PRED_LIST].append(_sr_prediction_list[0])

                if len(context[Context.METRICS_LIST]) == 0:

                    context[Context.METRICS_LIST] = pd.concat([pd.DataFrame([_metadata], index=[0])])
                else:
                    rows = len(context[Context.METRICS_LIST].index)
                    context[Context.METRICS_LIST] = pd.concat([context[Context.METRICS_LIST],  pd.DataFrame([_metadata], index=[rows])])

            # remove transient TOA arrays
            context[Context.METRICS_LIST].drop('toaBandMaArrayRaw', axis=1, inplace=True)
            # sr_metrics_list.drop('index', axis=1, inplace=True)
            context[Context.METRICS_LIST].reset_index()

                # sr_metrics_list = context[Context.METRICS_LIST]
                
            print(f'Ending ProcessingPool().amap() for toa: {str(bandPairIndicesList[j+1])}', flush=True)

        except BaseException as err:
                print('\simulateSurfaceReflectanceConcurrent processing failed - Error details: ', err)
                raise err

        return context[Context.PRED_LIST], context[Context.METRICS_LIST], context[Context.COMMON_MASK_LIST]


    def synchronized(wrapped):
        import functools
        import threading
        lock = threading.RLock()

        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            with lock:
                return wrapped(*args, **kwargs)

        return _wrapper

    @synchronized
    def _updateContext(self, context):
        import time
        print('\nUpdating context: ' +  str(context), flush=True)

    