import xarray as xr  # read rasters
import dask  # multi processsing library
import inspect
import numpy as np

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# module indices
# This class calculates remote sensing indices given xarray or numpy objects.
# Note: Most of our imagery uses the following set of bands.
# 8 band: ['CoastalBlue', 'Blue', 'Green', 'Yellow',
#          'Red', 'RedEdge', 'NIR1', 'NIR2']
# 4 band: ['Red', 'Green', 'Blue', 'NIR1', 'HOM1', 'HOM2']
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Module Methods
# -------------------------------------------------------------------------------

#########  ESRI:102001 experiment - pitkus-point-demo with 30m generated dynamically in GDAL,
#          CCDC edited (nodata & srs=ESRI:102001 using gdal_edit.py, and clipped using gdal_translate()
#          EVHR warped (reprojected & intersected) using warplib.memwarp_multi
#          Linear Regression - NUmpy modified for masked arrays (suppressed NaN, applied -9999 mask)
###########
b1_b_0 = -80.6135550938377
b1_b_1 = 3.193240132440603

b2_b_0 = -75.26841940681834
b2_b_1 = 1.915078674086289

b3_b_0 = -131.70956850193363
b3_b_1 = 1.8317816985941093

b4_b_0 = -330.1082466640355
b4_b_1 = 1.1325853043492222

def addindices(rastarr, bands, indices, factor=1.0) -> dask.array:
    """
     :param rastarr:
     :param indices:
     :param bands:
     :param factor:
     :return:
     """
    nbands = len(bands)  # get initial number of bands
    for indices_function in indices:  # iterate over each new band
        nbands += 1  # counter for number of bands

        # calculate band (indices)
        band, bandid = indices_function(rastarr, bands=bands, factor=factor)
        bands.append(bandid)  # append new band id to list of bands
        band.coords['band'] = [nbands]  # add band indices to raster
        rastarr = xr.concat([rastarr, band], dim='band')  # concat new band

    # update raster metadata, xarray attributes
    rastarr.attrs['scales'] = [rastarr.attrs['scales'][0]] * nbands
    rastarr.attrs['offsets'] = [rastarr.attrs['offsets'][0]] * nbands
    return rastarr, bands


# Difference Vegetation Index (DVI), type int16
def dvi(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: list with strings of bands in the raster
    :param factor: factor used for toa imagery
    :return: new band with DVI calculated
    """
    # 8 and 4 band imagery: DVI := NIR1 - Red
    NIR1, Red = bands.index('NIR1'), bands.index('Red')
    return ((data[NIR1, :, :] / factor) - (data[Red, :, :] / factor)
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "DVI"


# Normalized Difference Vegetation Index (NDVI)
# range from +1.0 to -1.0, type float64
def ndvi(data, bands, factor=1.0, vtype='float64') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with NDVI calculated
    """
    # 8 and 4 band imagery: NDVI := (NIR - Red) / (NIR + RED)
    NIR1, Red = bands.index('NIR1'), bands.index('Red')
    return (((data[NIR1, :, :] / factor) - (data[Red, :, :] / factor)) /
            ((data[NIR1, :, :] / factor) + (data[Red, :, :] / factor))
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "NDVI"


# Forest Discrimination Index (FDI), type int16
def fdi(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with FDI calculated
    """
    # 8 band imagery: FDI := NIR2 - (RedEdge + Blue)
    # 4 band imagery: FDI := NIR1 - (Red + Blue)
    NIR = bands.index('NIR2') if 'NIR2' in bands else bands.index('NIR1')
    Red = bands.index('RedEdge') if 'RedEdge' in bands else bands.index('Red')
    Blue = bands.index('Blue')
    return (data[NIR, :, :] - (data[Red, :, :] + data[Blue, :, :])
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "FDI"


# Shadow Index (SI), type int16
def si(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery:
    # SI := ((factor - Blue) * (factor - Green) * (factor - Red)) ** (1.0 / 3)
    Blue, Green = bands.index('Blue'), bands.index('Green')
    Red = bands.index('Red')
    return (((factor - data[Blue, :, :]) * (factor - data[Green, :, :]) *
            (factor - data[Red, :, :])) ** (1.0/3.0)
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "SI"


# Normalized Difference Water Index (DWI), type int16
def dwi(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with DWI calculated
    """
    # 8 and 4 band imagery: DWI := factor * (Green - NIR1)
    Green, NIR1 = bands.index('Green'), bands.index('NIR1')
    return (factor * (data[Green, :, :] - data[NIR1, :, :])
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "DWI"


# Normalized Difference Water Index (NDWI), type int16
def ndwi(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery: NDWI := factor * (Green - NIR1) / (Green + NIR1)
    Green, NIR1 = bands.index('Green'), bands.index('NIR1')
    return (factor * ((data[Green, :, :] - data[NIR1, :, :])
            / (data[Green, :, :] + data[NIR1, :, :]))
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "NDWI"


# Shadow Index (SI), type float64
def cs1(data, bands, factor=1.0, vtype='float64') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery: CS1 := (3. * NIR1) / (Blue + Green + Red)
    NIR1, Blue = bands.index('NIR1'), bands.index('Blue')
    Green, Red = bands.index('Green'), bands.index('Red')
    return ((3.0 * (data[NIR1, :, :]/factor)) / (data[Blue, :, :]
            + data[Green, :, :] + data[Red, :, :])
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "CS1"


# Shadow Index (SI)
def cs2(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with CS2 calculated
    """
    # 8 and 4 band imagery: CS2 := (Blue + Green + Red + NIR1) / 4.
    NIR1, Blue = bands.index('NIR1'), bands.index('Blue')
    Green, Red = bands.index('Green'), bands.index('Red')
    return ((data[Blue, :, :] + data[Green, :, :] + data[Red, :, :]
            + data[NIR1, :, :]) / 4.0
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "CS2"

# Linear Regression(LR), type int16
def lr_(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    # 8 and 4 band imagery:
    slope = 0.3580659703162858
    yInt = 865.3930757956043

    #y = ee.Number(x).multiply(slope).add(yInt);
    # LR := ((factor - Blue) * (factor - Green) * (factor - Red)) ** (1.0 / 3)
    NIR = bands.index('b4')
    return ((data[NIR, :, :] * slope) + yInt
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "LR"

# Linear Regression(LR), type int16
def lr(data, bands, index, name, factor=1.0, vtype='int16') -> dask.array:
    """
    """
    # 8 and 4 band imagery:
    slope = 0.3580659703162858
    yInt = 865.3930757956043

    #y = ee.Number(x).multiply(slope).add(yInt);
    # LR := ((factor - Blue) * (factor - Green) * (factor - Red)) ** (1.0 / 3)
    band = bands.index(index)
    xform = ((data[band, :, :] * slope) + yInt
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), name
    return (xform)

# Linear Regression(LR), type int16
def __lr(yInt, slope, data, bands, index, name, factor=1.0, vtype='int16') -> dask.array:
    """
    """
    #y = ee.Number(x).multiply(slope).add(yInt);
    # LR := ((factor - Blue) * (factor - Green) * (factor - Red)) ** (1.0 / 3)
    band = bands.index(index)
    xform = ((data[band, :, :] * slope) + yInt
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), name
    return (xform)

# b1 = Blue, type int16
def blue(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    return (lr(b1_b_0, b1_b_1, data, bands, "blue", "Blue", factor, vtype))

# b2 = Green, type int16
def green(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    return (lr(b2_b_0, b2_b_1, data, bands, "green", "Green", factor, vtype))

# b3 = Red, type int16
def red(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    return (lr(b3_b_0, b3_b_1, data, bands, "red", "Red", factor, vtype))

# b4 = NIR, type int16
def nir(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    return (lr(b4_b_0, b4_b_1, data, bands, "nir", "NIR", factor, vtype))

# Linear Regression(LR), type int16
def lr(data, bands, coefficients, fname, factor=1.0, vtype='int16') -> dask.array:
    """
    """
    index = int(fname[1])
    slope, yInt = coefficients[index][0], coefficients[index][1]

    #y = ee.Number(x).multiply(slope).add(yInt);
    band = bands.index(fname)
    print ("Name BandIndex Slope Yint: ", fname, band, slope, yInt)

    xform = ((data[band, :, :] * slope) + yInt
             ).expand_dims(dim="band", axis=0).fillna(-9999).astype(vtype), fname
#    ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), fname
    return (xform)

def getBandIndex(data, bandNames, fname) :
    """
    TODO: Replace logic with real mapping...
    Temporary slop to deal with lack of metadata in WV files
    """
    index = 0
    if (data.shape[0] > 4) :
        if (fname == 'blue') : index = 1
        elif (fname == 'green'): index = 2
        elif (fname == 'red'): index = 4
        elif (fname == 'nir'): index = 6
    else:
        # assumes order is honored by EVHR
        index = bandNames.index(fname)

    return index

# Linear Regression(LR), type int16
def _map(data, coefficients, bandNames, fname, factor=1.0, vtype='int16') -> dask.array:
    """
    """
    band = int(getBandIndex(data, bandNames, fname))
    nameIndex = int(bandNames.index(fname))
    slope, yInt = coefficients[nameIndex+1][0], coefficients[nameIndex+1][1]

    #y = ee.Number(x).multiply(slope).add(yInt);
    print ("Name BandIndex Slope Yint: ", fname, band, slope, yInt)
    bandArray = (data[band, :, :] )
#    print(f'bandArray.mean =  {bandArray.mean} ')
    nbandArray = bandArray.as_numpy()
    print(f'nbandArray.average =  {np.average(nbandArray)} index[0][0] {nbandArray.values[0][0]} index[500][500] {nbandArray.values[500][500]}')

    newData = ((bandArray) * slope) + yInt
    nnewData = newData.as_numpy()
    print(f'nnewData.average =  {np.average(nnewData)} index[0][0] {nnewData.values[0][0]} index[500][500] {nnewData.values[500][500]}')

    xform = (newData
#             ).expand_dims(dim="band", axis=0).fillna(-9999).astype(vtype), fname
    ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), fname
#    print(f'xform.mean =  {xform[0].mean} ')
    nxformArray = xform[0].as_numpy()
#    print(f'xform.average = {np.average(nxformArray)} ')
#    print(f'nxformArray.average =  {np.average(nxformArray)} index[0][0] {nxformArray.values[0][0]} index[500][500] {nxformArray.values[500][500]}')

    import matplotlib.pyplot as plt
    x = nbandArray.values[0]
    y = nbandArray.values[1]
    x2 = x[y > -9999]
    y2 = y[x > -9999]
    plt.plot(x2, y2)


    _x = nnewData.values[0]
    _y= nnewData.values[1]
    _x2 = _x[_y > -9999]
    _y2 = _y[_x > -9999]
    plt.plot(_x2, _y2)

#    a = np.arange(5)
    hist, bin_edges = np.histogram(bandArray, density=True)
#    print(f'bandArray hist {hist} sum {hist.sum()}')

    hist1, bin_edges1 = np.histogram(newData, density=True)
#    print(f'newData hist {hist1} sum {hist1.sum()}')

#    np.sum(hist * np.diff(bin_edges))
#    rng = np.random.RandomState(10)  # deterministic random data
#    a = np.hstack((rng.normal(size=1000),rng.normal(loc=5, scale=2, size=1000)))
#    _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
#    plt.title("Histogram with 'auto' bins")

    return (xform)


#  See https://code.earthengine.google.com/?accept_repo=users%2Fwiell%2FtemporalSegmentation&scriptPath=users%2Fwiell%2FtemporalSegmentation%3AtemporalSegmentation
# blue_offset: -584.2706236346759
# blue_scale: 0.8080896368949587
# green_offset: -554.5582699867354
# green_scale: 1.092750480442165
# nir_offset: -299.0741382381461
# nir_scale: 1.184029871713611
# red_offset: -378.46913058303215
# red_scale: 1.115311673152504

    # _overrides[1][1] = -584.2706236346759
    # ___ovverrides[1][1][1,(0.8080896368949587, -584.2706236346759)], #blur
    #               (1.092750480442165, -554.5582699867354), #green
    #               (1.184029871713611, -299.0741382381461), #nir
    #               (1.115311673152504, -378.46913058303215)] #red
    #


# Linear Regression(LR), type int16
def map(data, coefficients, bandNames, fname, factor=1.0, vtype='int16') -> dask.array:
    """
    """

# 2011

# got below from GEE, so switch bottom two
# https://code.earthengine.google.com/?accept_repo=users%2Fwiell%2FtemporalSegmentation&scriptPath=users%2Fwiell%2FtemporalSegmentation%3AtemporalSegmentation
    # want ['blue', 'green', 'red', 'nir']
    # coefficients[1]= (0.8080896368949587, -584.2706236346759)
    # coefficients[2]= (1.092750480442165, -554.5582699867354)
    #
    # coefficients[4]= (1.184029871713611, -299.0741382381461)  #### SWITCHED NIR & RED
    # coefficients[3]= (1.115311673152504, -378.46913058303215)

# Goofy numbers from numpy LR
#            lr = SimpleLinearRegression(ccdcMaArray, evhrMaArray)
#1 = {tuple} <class 'tuple'>: (-1.3899176546960916, 2.6542156715596263)
#2 = {tuple} <class 'tuple'>: (-0.7170513164908243, 1.6242873991639362)
#3 = {tuple} <class 'tuple'>: (0.1774494399148807, 1.3396771789639992)
#4 = {tuple} <class 'tuple'>: (0.04033702255310345, 0.984404006211036)

#    lr = SimpleLinearRegression(evhrMaArray, ccdcMaArray)
#1 = {tuple} <class 'tuple'>: (-0.004990721986189328, 0.3771709090197055)
#2 = {tuple} <class 'tuple'>: (0.0013340963275823015, 0.6160490647969467)
#3 = {tuple} <class 'tuple'>: (-0.009759089702924939, 0.7463079203277814)
#4 = {tuple} <class 'tuple'>: (0.0012908794269606005, 1.0158277887434783)

    nameIndex = int(bandNames.index(fname))
    slope, yInt = coefficients[nameIndex+1][0], coefficients[nameIndex+1][1]

    #y = ee.Number(x).multiply(slope).add(yInt);
    band = int(getBandIndex(data, bandNames, fname))
    print ("Name BandIndex Slope Yint: ", fname, band, slope, yInt)
    bandArray = (data[band, :, :] )
#    print(f'bandArray.mean =  {bandArray.mean} ')
    nbandArray = bandArray.as_numpy()
    print(f'nbandArray.average =  {np.average(nbandArray)} index[0][0] {nbandArray.values[0][0]} index[500][500] {nbandArray.values[500][500]}')

    newData = ((bandArray) * slope) + yInt
    nnewData = newData.as_numpy()
    print(f'nnewData.average =  {np.average(nnewData)} index[0][0] {nnewData.values[0][0]} index[500][500] {nnewData.values[500][500]}')

    xform = (newData
#             ).expand_dims(dim="band", axis=0).fillna(-9999).astype(vtype), fname
    ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), fname
#    print(f'xform.mean =  {xform[0].mean} ')
    nxformArray = xform[0].as_numpy()
#    print(f'xform.average = {np.average(nxformArray)} ')
#    print(f'nxformArray.average =  {np.average(nxformArray)} index[0][0] {nxformArray.values[0][0]} index[500][500] {nxformArray.values[500][500]}')

    # import matplotlib.pyplot as plt
    # x = nbandArray.values[0]
    # y = nbandArray.values[1]
    # x2 = x[y > -9999]
    # y2 = y[x > -9999]
    # plt.plot(x2, y2)
    #
    #
    # _x = nnewData.values[0]
    # _y= nnewData.values[1]
    # _x2 = _x[_y > -9999]
    # _y2 = _y[_x > -9999]
    # plt.plot(_x2, _y2)

#    a = np.arange(5)
    hist, bin_edges = np.histogram(bandArray, density=True)
#    print(f'bandArray hist {hist} sum {hist.sum()}')

    hist1, bin_edges1 = np.histogram(newData, density=True)
#    print(f'newData hist {hist1} sum {hist1.sum()}')

#    np.sum(hist * np.diff(bin_edges))
#    rng = np.random.RandomState(10)  # deterministic random data
#    a = np.hstack((rng.normal(size=1000),rng.normal(loc=5, scale=2, size=1000)))
#    _ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
#    plt.title("Histogram with 'auto' bins")

    return (xform)

# Normalized Difference Water Index (NDWI), type int16
def _ndwi(data, bands, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery: NDWI := factor * (Green - NIR1) / (Green + NIR1)
    Green, NIR1 = bands.index('Green'), bands.index('NIR1')
    return (factor * ((data[Green, :, :] - data[NIR1, :, :])
            / (data[Green, :, :] + data[NIR1, :, :]))
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "NDWI"

# b1 = Blue, type int16
def b1(data, bands, coefficients, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    xform = (lr(data, bands, coefficients, inspect.currentframe().f_code.co_name, factor, vtype))
    return xform

# b2 = Green, type int16
def b2(data, bands, coefficients, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    xform = (lr(data, bands, coefficients, inspect.currentframe().f_code.co_name, factor, vtype))
    return xform

# b3 = Red, type int16
def b3(data, bands, coefficients, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    xform = (lr(data, bands, coefficients, inspect.currentframe().f_code.co_name, factor, vtype))
    return xform

# b4 = NIR, type int16
def b4(data, bands, coefficients, factor=1.0, vtype='int16') -> dask.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with LR calculated
    """
    xform = (lr(data, bands, coefficients, inspect.currentframe().f_code.co_name, factor, vtype))
    return xform

