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

# SR-Lite dependencies

# --------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------

def processBands(context, warp_ds_list, bandNamePairList,
                 bandPairIndicesList, fn_list,
                 r_fn_cloudmask_warp,
                 plotLib, rasterLib):

    import numpy.ma as ma

    ccdc_warp_ds = warp_ds_list[0]
    evhr_warp_ds = warp_ds_list[1]

    ########################################
    # ### PREPARE CLOUDMASK
    # After retrieving the masked array from the warped cloudmask, further reduce it by suppressing the one ("1") value pixels
    ########################################
    plotLib.trace('bandPairIndicesList: ' + str(bandPairIndicesList))
    numBandPairs = len(bandPairIndicesList)
    warp_ma_masked_band_series = [numBandPairs]
    sr_prediction_list = [numBandPairs]

    #  Get Masked array from warped Cloudmask - assumes only 1 band in mask to be applied to all
    cloudmaskWarpExternalBandMaArray = iolib.fn_getma(r_fn_cloudmask_warp, 1)
    plotLib.trace(f'\nBefore Mask -> cloudmaskWarpExternalBandMaArray')
    plotLib.trace(f'cloudmaskWarpExternalBandMaArray hist: {np.histogram(cloudmaskWarpExternalBandMaArray)}')
    plotLib.trace(f'cloudmaskWarpExternalBandMaArray shape: {cloudmaskWarpExternalBandMaArray.shape}')
    count_non_masked = ma.count(cloudmaskWarpExternalBandMaArray)
    count_masked = ma.count_masked(cloudmaskWarpExternalBandMaArray)
    plotLib.trace(f'cloudmaskWarpExternalBandMaArray ma.count (masked)=' + str(count_non_masked))
    plotLib.trace(f'cloudmaskWarpExternalBandMaArray ma.count_masked (non-masked)=' + str(count_masked))
    plotLib.trace(
        f'cloudmaskWarpExternalBandMaArray total count (masked + non-masked)=' + str(count_masked + count_non_masked))
    plotLib.trace(f'cloudmaskWarpExternalBandMaArray max=' + str(cloudmaskWarpExternalBandMaArray.max()))
    plotLib.plot_combo_array(cloudmaskWarpExternalBandMaArray, figsize=(14, 7), title='cloudmaskWarpExternalBandMaArray')

    # Create a mask where the pixel values equal to 'one' are suppressed because these correspond to clouds
    plotLib.trace(f'\nAfter Mask == 1.0 (sum should be 0 since all ones are masked -> cloudmaskWarpExternalBandMaArray')
    cloudmaskWarpExternalBandMaArrayMasked = np.ma.masked_where(cloudmaskWarpExternalBandMaArray == 1.0,
                                                                cloudmaskWarpExternalBandMaArray)
    plotLib.trace(f'cloudmaskWarpExternalBandMaArrayMasked hist: {np.histogram(cloudmaskWarpExternalBandMaArrayMasked)}')
    plotLib.trace(f'cloudmaskWarpExternalBandMaArrayMasked shape: {cloudmaskWarpExternalBandMaArrayMasked.shape}')
    count_non_masked = ma.count(cloudmaskWarpExternalBandMaArrayMasked)
    count_masked = ma.count_masked(cloudmaskWarpExternalBandMaArrayMasked)
    plotLib.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count (masked)=' + str(count_non_masked))
    plotLib.trace(f'cloudmaskWarpExternalBandMaArrayMasked ma.count_masked (non-masked)=' + str(count_masked))
    plotLib.trace(f'cloudmaskWarpExternalBandMaArrayMasked total count (masked + non-masked)=' + str(
        count_masked + count_non_masked))
    plotLib.trace(f'cloudmaskWarpExternalBandMaArrayMasked max=' + str(cloudmaskWarpExternalBandMaArrayMasked.max()))
    plotLib.plot_combo_array(cloudmaskWarpExternalBandMaArrayMasked, figsize=(14, 7),
                  title='cloudmaskWarpExternalBandMaArrayMasked')

    #         # Get Cloud mask (assumes 1 band per scene)
    #     cloudmaskArray = iolib.fn_getma(r_fn_cloudmask, 1)
    # #    cloudmaskArray = iolib.ds_getma(warp_ds_list[2], 1)
    #     plotLib.trace(f'cloudmaskArray array shape: {cloudmaskArray.shape}')
    #     cloudmaskMaArray = np.ma.masked_where(cloudmaskArray == 1, cloudmaskArray)
    #     plotLib.trace(f'cloudmaskMaArray array shape: {cloudmaskMaArray.shape}')

    minWarning = 0
    firstBand = True
    threshold = False
    if (threshold == True):
        # Apply range of -100 to 200 for Blue Band pixel mask and apply to each band
        evhrBandMaArrayThresholdMin = -100
        #    evhrBandMaArrayThresholdMax = 10000
        evhrBandMaArrayThresholdMax = 2000
        plotLib.trace(' evhrBandMaArrayThresholdMin = ' + str(evhrBandMaArrayThresholdMin))
        plotLib.trace(' evhrBandMaArrayThresholdMax = ' + str(evhrBandMaArrayThresholdMax))
        minWarning = evhrBandMaArrayThresholdMin

    ########################################
    # ### FOR EACH BAND PAIR,
    # now, each input should have same exact dimensions, grid, projection. They ony differ in their values (CCDC is surface reflectance, EVHR is TOA reflectance)
    ########################################
    for bandPairIndex in range(0, numBandPairs - 1):

        plotLib.trace('=>')
        plotLib.trace('====================================================================================')
        plotLib.trace('============== Start Processing Band #' + str(bandPairIndex + 1) + ' ===============')
        plotLib.trace('====================================================================================')

        # Retrieve band pair
        bandPairIndices = bandPairIndicesList[bandPairIndex + 1]

        # Get 30m CCDC Masked Arrays
        # ccdcBandMaArray = iolib.ds_getma(ccdc_warp_ds, 1)
        # evhrBandMaArray = iolib.ds_getma(evhr_warp_ds, 2)
        ccdcBandMaArray = iolib.ds_getma(warp_ds_list[0], bandPairIndices[0])
        evhrBandMaArray = iolib.ds_getma(warp_ds_list[1], bandPairIndices[1])

        #  Create single mask for all bands based on Blue-band threshold values
        #  Assumes Blue-band is first indice pair, so collect mask on 1st iteration only.
        if (threshold == True):
            if (firstBand == True):
                evhrBandMaArray = rasterLib.applyThreshold(evhrBandMaArrayThresholdMin, evhrBandMaArrayThresholdMax,
                                                 evhrBandMaArray)
                firstBand = False

        #  Create a common mask that intersects the CCDC, EVHR, and Cloudmask - this will then be used to correct the input EVHR & CCDC
        warp_ma_band_list_all = [ccdcBandMaArray, evhrBandMaArray, cloudmaskWarpExternalBandMaArrayMasked]
        common_mask_band_all = malib.common_mask(warp_ma_band_list_all)
        # plotLib.trace(f'common_mask_band_all hist: {np.histogram(common_mask_band_all)}')
        # plotLib.trace(f'common_mask_band_all array shape: {common_mask_band_all.shape}')
        # plotLib.trace(f'common_mask_band_all array sum: {common_mask_band_all.sum()}')
        # # plotLib.trace(f'common_mask_band_data_only array count: {common_mask_band_all.count}')
        # plotLib.trace(f'common_mask_band_all array max: {common_mask_band_all.max()}')
        # plotLib.trace(f'common_mask_band_all array min: {common_mask_band_all.min()}')
        # # plot_combo(common_mask_band_all, figsize=(14,7), title='common_mask_band_all')
        # count_non_masked = ma.count(int(common_mask_band_all))
        # count_masked = ma.count_masked(common_mask_band_all)
        # plotLib.trace(f'common_mask_band_all ma.count (masked)=' + str(count_non_masked))
        # plotLib.trace(f'common_mask_band_all ma.count_masked (non-masked)=' + str(count_masked))
        # plotLib.trace(f'common_mask_band_all total count (masked + non-masked)=' + str(count_masked + count_non_masked))

        # Apply the 3-way common mask to the CCDC and EVHR bands
        warp_ma_masked_band_list = [np.ma.array(ccdcBandMaArray, mask=common_mask_band_all),
                                    np.ma.array(evhrBandMaArray, mask=common_mask_band_all)]

        # Check the mins of each ma - they should be greater than 0
        for j, ma in enumerate(warp_ma_masked_band_list):
            j = j + 1
            if (ma.min() < minWarning):
                plotLib.trace("Warning: Masked array values should be larger than " + str(minWarning))
        #            exit(1)
        plotLib.plot_maps(warp_ma_masked_band_list, fn_list, figsize=(10, 5),
                     title=str(bandNamePairList[bandPairIndex]) + ' Reflectance (%)')
        plotLib.plot_histograms(warp_ma_masked_band_list, fn_list, figsize=(10, 3),
                           title=str(bandNamePairList[bandPairIndex]) + " BAND COMMON ARRAY")

        warp_ma_masked_band_series.append(warp_ma_masked_band_list)

        ########################################
        # ### WARPED MASKED ARRAY WITH COMMON MASK, DATA VALUES ONLY
        # CCDC SR is first element in list, which needs to be the y-var: b/c we are predicting SR from TOA ++++++++++[as per PM - 01/05/2022]
        ########################################
        ccdc_sr_band = warp_ma_masked_band_list[0].ravel()
        evhr_toa_band = warp_ma_masked_band_list[1].ravel()

        ccdc_sr_data_only_band = ccdc_sr_band[ccdc_sr_band.mask == False]
        evhr_toa_data_only_band = evhr_toa_band[evhr_toa_band.mask == False]

        # Perform regression fit based on model type!
        if (context[Context.REGRESSION_MODEL] == Context.REGRESSOR_ROBUST):
            model_data_only_band = HuberRegressor().fit(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
        else:
            model_data_only_band = LinearRegression().fit(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)

        plotLib.trace(str(bandNamePairList[bandPairIndex]) + '= > intercept: ' + str(
            model_data_only_band.intercept_) + ' slope: ' + str(model_data_only_band.coef_) + ' score: ' +
                 str(model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)))
        plotLib.plot_fit(evhr_toa_data_only_band, ccdc_sr_data_only_band, model_data_only_band.coef_[0],
                    model_data_only_band.intercept_)

        ########################################
        # #### Apply the model to the original EVHR (2m) to predict surface reflectance
        ########################################
        plotLib.trace(f'Applying model to {str(bandNamePairList[bandPairIndex])} in file {os.path.basename(fn_list[1])}')
        plotLib.trace(f'Input masked array shape: {evhrBandMaArray.shape}')

        score = model_data_only_band.score(evhr_toa_data_only_band.reshape(-1, 1), ccdc_sr_data_only_band)
        plotLib.trace(f'R2 score : {score}')

        # Get 2m EVHR Masked Arrays
        evhrBandMaArrayRaw = iolib.fn_getma(fn_list[1], bandPairIndices[1])
        sr_prediction_band = model_data_only_band.predict(evhrBandMaArrayRaw.ravel().reshape(-1, 1))
        plotLib.trace(f'Post-prediction shape : {sr_prediction_band.shape}')

        # Return to original shape and apply original mask
        orig_dims = evhrBandMaArrayRaw.shape
        evhr_sr_ma_band = np.ma.array(sr_prediction_band.reshape(orig_dims), mask=evhrBandMaArrayRaw.mask)

        # Check resulting ma
        plotLib.trace(f'Final masked array shape: {evhr_sr_ma_band.shape}')
        #    plotLib.trace('evhr_sr_ma=\n' + str(evhr_sr_ma_band))

        ########### save prediction #############
        sr_prediction_list.append(evhr_sr_ma_band)

        ########################################
        ##### Compare the before and after histograms (EVHR TOA vs EVHR SR)
        ########################################
        evhr_pre_post_ma_list = [evhrBandMaArrayRaw, evhr_sr_ma_band]
        compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']

        plotLib.plot_histograms(evhr_pre_post_ma_list, fn_list, figsize=(5, 3),
                           title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR")
        plotLib.plot_maps(evhr_pre_post_ma_list, compare_name_list, figsize=(10, 50))

        ########################################
        ##### Compare the original CCDC histogram with result (CCDC SR vs EVHR SR)
        ########################################
        #     ccdc_evhr_srlite_list = [ccdc_warp_ma, evhr_sr_ma_band]
        #     compare_name_list = ['CCDC SR', 'EVHR SR-Lite']

        #     plotLib.plot_histograms(ccdc_evhr_srlite_list, fn_list, figsize=(5, 3),
        #                        title=str(bandNamePairList[bandPairIndex]) + " CCDC SR vs EVHR SR", override=override)
        #     plotLib.plot_maps(ccdc_evhr_srlite_list, compare_name_list, figsize=(10, 50), override=override)

        ########################################
        ##### Compare the original EVHR TOA histogram with result (EVHR TOA vs EVHR SR)
        ########################################
        evhr_srlite_delta_list = [evhr_pre_post_ma_list[1], evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]]
        compare_name_list = ['EVHR TOA', 'EVHR SR-Lite']
        plotLib.plot_histograms(evhr_srlite_delta_list, fn_list, figsize=(5, 3),
                           title=str(bandNamePairList[bandPairIndex]) + " EVHR TOA vs EVHR SR DELTA ")
        plotLib.plot_maps([evhr_pre_post_ma_list[1],
                      evhr_pre_post_ma_list[1] - evhr_pre_post_ma_list[0]],
                     [compare_name_list[1],
                      str(bandNamePairList[bandPairIndex]) + ' Difference: TOA-SR-Lite'], (10, 50),
                     cmap_list=['RdYlGn', 'RdBu'])
        print(f"Finished with {str(bandNamePairList[bandPairIndex])} Band")

    return sr_prediction_list

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

    # Debug levels:  0-no debug, 1-trace, 2-visualization, 3-detailed diagnostics
    debug_level = int(context[Context.DEBUG_LEVEL])

    # Get handles to plot and raster classes
    plotLib = contextClazz.getPlotLib()
    rasterLib = RasterLib(debug_level, plotLib)
    cogname = "No File"

    for context[Context.FN_TOA] in sorted(Path(context[Context.DIR_TOA]).glob("*.tif")):
#    for context[Context.FN_TOA] in (Path(context[Context.DIR_TOA]).glob("*.tif")):

        try:
            # Generate file names based on incoming EVHR file and declared suffixes
            context = contextClazz.getFileNames(str(context[Context.FN_TOA]).rsplit("/", 1), context)

            # Get attributes of raw EVHR tif and create plot - assumes same root name suffixed by "-toa.tif")
            plotLib.trace('\nEVHR file=' + str(context[Context.FN_TOA]))
            rasterLib.getProjection(str(context[Context.FN_TOA]), "EVHR Combo Plot")

            # Get attributes of raw CCDC tif and create plot - assumes same root name suffixed by '-ccdc.tif')
            plotLib.trace('\nCCDC file=' + str(context[Context.FN_CCDC]))
            rasterLib.getProjection(str(context[Context.FN_CCDC]), "CCDC Combo Plot")

            # Get attributes of raw cloudmask tif and create plot - assumes same root name suffixed by '-toa_pred.tif')
            plotLib.trace('\nCloudmask file=' + str(context[Context.FN_CLOUDMASK]))
            rasterLib.getProjection(str(context[Context.FN_CLOUDMASK]), "Cloudmask Combo Plot")

            #  Warp cloudmask to attributes of EVHR - suffix root name with '-toa_pred_warp.tif')
            plotLib.trace('\nCloudmask Warp=' + str(context[Context.FN_WARP]))
            context[Context.FN_SRC] = str(context[Context.FN_CLOUDMASK])
            context[Context.FN_DEST] = str(context[Context.FN_WARP])
            context[Context.TARGET_ATTR] = str(context[Context.FN_TOA])
            rasterLib.downscale(context)
            rasterLib.getProjection(str(context[Context.FN_WARP]), "Cloudmask Warp Combo Plot")

            # Validate that input band name pairs exist in EVHR & CCDC files
            context[Context.FN_LIST] = [str(context[Context.FN_CCDC]), str(context[Context.FN_TOA])]
            bandPairIndicesList = rasterLib.getBandIndices(context)

            # Get the common pixel intersection values of the EVHR & CCDC files
            warp_ds_list, warp_ma_list = rasterLib.getIntersection(context[Context.FN_LIST])

            # Perform regression to capture coefficients from intersected pixels and apply to 2m EVHR
            sr_prediction_list = processBands(context, warp_ds_list, list(ast.literal_eval(context[Context.LIST_BAND_PAIRS])), bandPairIndicesList,
                                           context[Context.FN_LIST], context[Context.FN_WARP], plotLib, rasterLib)

            # Create COG image from stack of processed bands
            cogname = rasterLib.createImage(context,
                str(context[Context.FN_TOA]), len(bandPairIndicesList), sr_prediction_list,
                str(context[Context.FN_PREFIX]),
                list(ast.literal_eval(context[Context.LIST_BAND_PAIRS])),
                context[Context.DIR_OUTPUT])

        except FileNotFoundError as exc:
            print(exc);
        except BaseException as err:
            print('Run abended.  Error: ', err)
            sys.exit(1)

#        break;

    print("\nTotal Elapsed Time for " + cogname + ': ',
           (time.time() - start_time) / 60.0)  # time in min

if __name__ == "__main__":
    main()
