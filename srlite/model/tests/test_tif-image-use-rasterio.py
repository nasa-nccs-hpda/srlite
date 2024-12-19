
############
import rasterio
from matplotlib import pyplot
from lr.SimpleLinearRegression import SimpleLinearRegression
from osgeo import gdal, osr
from osgeo import gdal_array
from pygeotools.lib import iolib, warplib, malib
import numpy as np
from core.model.SystemCommand import SystemCommand
from os.path import exists

# -------- THE MAGIC COMBINATION V2 --------
pitkusPoint = [-2927850, 4063820, -2903670, 4082450]
bbox = pitkusPoint
pitkusPointStr = ['-2927850', '4063820', '-2903670', '4082450']
bboxStr = pitkusPointStr
dstSRS = 'ESRI:102001'
# dstSRS = 'EPSG:4326'
dstNodata = -9999
driver = 'MEM'  # process in memory
xres = 30
yres = 30

def save_rasters(array, path, dst_filename):
    """ Save the final multiband array based on an existing raster """

    example = gdal.Open(path)
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(dst_filename, x_pixels,
                            y_pixels, bands, gdal.GDT_Float64)

    geotrans = example.GetGeoTransform()  # get GeoTranform from existed 'data0'
    proj = example.GetProjection()  # you can get from a exsited tif or import
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    for b in range(bands):
        dataset.GetRasterBand(b + 1).SetNoDataValue(dstNodata)
        dataset.GetRasterBand(b + 1).WriteArray(array[b, :, :])

    dataset.FlushCache()


def save_rasters_memory(array, path):
    """ Save the final multiband array based on an existing raster """

    example = gdal.Open(path)
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', x_pixels,
                            y_pixels, bands, gdal.GDT_Float64)

    geotrans = example.GetGeoTransform()  # get GeoTranform from existed 'data0'
    proj = example.GetProjection()  # you can get from a exsited tif or import
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    for b in range(bands):
#        dataset.GetRasterBand(b + 1).SetNoDataValue(dstNodata)
        dataset.GetRasterBand(b + 1).WriteArray(array[b, :, :])

    dataset.FlushCache()
    return dataset

#  Specify input files and intermediate paths
pathRoot = '/home/centos/srlite/LR/srlite-workflow-113021-demo/'
ccdcFilePrefix = 'CCDC-pitkus-point-epsg102001'
evhrFilePrefix = 'WV02_20110818_M1BS_103001000CCC9000-toa'

ccdcFn = pathRoot + ccdcFilePrefix + '.tif'
ccdcEditNoNanFn = pathRoot + '_' + ccdcFilePrefix + '-edit-nonan.tif'
ccdcEditNoNanClipFn = pathRoot + '_' + ccdcFilePrefix + '-bbox.tif'
ccdcEditNoNanClipResFn = pathRoot + '_' + ccdcFilePrefix + '-bbox-res.tif'

evhrHighResFn = pathRoot + evhrFilePrefix + '.tif'
evhrHighResIntersectionFn = pathRoot + evhrFilePrefix + '-intersection.tif'

# deprecated
evhrHighResClipFn = pathRoot + '_WV02_20110818_M1BS_103001000CCC9000-toa-bbox.tif'
evhrHighResClipResFn = pathRoot + '_WV02_20110818_M1BS_103001000CCC9000-toa-bbox-res.tif'

# Determine whether to recreate intermediate files
ccdc_transformation_finished = exists(ccdcEditNoNanClipResFn)
evhr_transformation_finished = exists(evhrHighResIntersectionFn)

# Transform CCDC (if needed)
if ccdc_transformation_finished == False:
    # initialize run (replace updated ccdc file with fresh copy)
    command = 'cp /home/centos/srlite/LR/navaranak-demo/CCDC-pitkus-point-epsg102001-orig.tif /home/centos/srlite/LR/navaranak-demo/CCDC-pitkus-point-epsg102001.tif'
    SystemCommand(command)

    # Use gdal_edit (via ILAB core SystemCommand) to convert GEE CCDC output to proper projection ESRI:102001 and set NoData value in place
    command = 'gdal_edit.py -a_nodata -9999 -a_srs ESRI:102001 ' + ccdcFn
    SystemCommand(command)

    # Use gdal to load original CCDC datafile as 3-dimensional array
    ccdcEditArr = gdal_array.LoadFile(ccdcFn)

    # Replace 'nan' values with nodata value (i.e., -9999) and store to disk
    ccdcEditArrNoNan = np.nan_to_num(ccdcEditArr, copy=True, nan=dstNodata)
    clippedDs = save_rasters(ccdcEditArrNoNan, ccdcFn, ccdcEditNoNanFn)

    #  Apply bounding box and scale to CCDC data - API barked when I tried both in same gdal_translate() call
    command = \
        'gdal_translate -a_srs ESRI:102001 -a_ullr -2927850 4063820 -2903670 4082450 -a_nodata ' \
        + str(dstNodata) + ' ' + ccdcEditNoNanFn + ' ' + ccdcEditNoNanClipFn
    SystemCommand(command)
    command = 'gdal_translate -tr 30 30 ' + ccdcEditNoNanClipFn + ' ' + ccdcEditNoNanClipResFn
    SystemCommand(command)

# Transform EVHR (if needed)
ccdcEditNoNanClipResDs = gdal.Open(ccdcEditNoNanClipResFn, gdal.GA_Update)
evhrHighResFnDs = gdal.Open(evhrHighResFn, gdal.GA_Update)

if evhr_transformation_finished == False:

    #  Call memwarp to get the 'intersection' of the CCDC & EVHR datasets
    ds_list = warplib.memwarp_multi(
        [ccdcEditNoNanClipResDs, evhrHighResFnDs], res='max',
        #    [ccdcEditNoNanClipResDs, evhrHighResClipResDs], res='max',
        extent='intersection', t_srs='first', r='average', dst_ndv=dstNodata)

    # Name the datesets
    ccdcDsAfterMemwarp = ds_list[0]
    evhrDsAfterMemwarp = ds_list[1]

    # Store the 'intersection' file to disk (assume that ccdcDsAfterMemwarp is unchanged)
    evhrDsAfterMemwarpArr = gdal_array.DatasetReadAsArray(evhrDsAfterMemwarp)
    save_rasters(evhrDsAfterMemwarpArr, evhrHighResFn, evhrHighResIntersectionFn)

else:

    # Load the 'intersection' file to disk
    ccdcDsAfterMemwarp = ccdcEditNoNanClipResDs
    evhrDsAfterMemwarp = gdal.Open(evhrHighResIntersectionFn, gdal.GA_Update)

# add transformed WV Bands
# B1(Blue): 450 - 510
# B2(Green): 510 - 580
# B3(Red): 655 - 690
# B4(NIR): 780 - 920
    # Coefficients
    #ccdcBands
    #ccdcBand = ccdcDsAfterMemwarp.GetRasterBand("nir").ReadAsArray()
    #evhrBand = evhrDsAfterMemwarp.GetRasterBand("b4").ReadAsArray()

# Get Numpy array from dataset for linear regression (per band)
for b in range(ccdcDsAfterMemwarp.RasterCount):
    ccdcBand = ccdcDsAfterMemwarp.GetRasterBand(b + 1).ReadAsArray()
    ndv = ccdcDsAfterMemwarp.GetRasterBand(b + 1).GetNoDataValue()
    print(ndv)
#    ccdcBand.SetNoDataValue(dstNodata)
#    ndv = ccdcBand.GetNoDataValue()
#    ccdcBandArr = ccdcBand.ReadAsArray()

    evhrDsAfterMemwarp.GetRasterBand(b + 1).SetNoDataValue(dstNodata)
    evhrBand = evhrDsAfterMemwarp.GetRasterBand(b + 1).ReadAsArray()
    ndv = evhrDsAfterMemwarp.GetRasterBand(b + 1).GetNoDataValue()
    print(ndv)
#    evhrBand.SetNoDataValue(dstNodata)
#    ndv = evhrBand.GetNoDataValue()
#    evhrBandArr = ccdcBand.ReadAsArray()

    lr = SimpleLinearRegression(ccdcBand, evhrBand)
    lr.run()

exit();

# -------- THE MAGIC COMBINATION V2 --------
# -------- THE MAGIC COMBINATION --------
def save_rasters_memory(array, path):
    """ Save the final multiband array based on an existing raster """

    example = gdal.Open(path)
    x_pixels = array.shape[2]  # number of pixels in x
    y_pixels = array.shape[1]  # number of pixels in y
    bands = array.shape[0]
    driver = gdal.GetDriverByName('MEM')
    dataset = driver.Create('', x_pixels,
                            y_pixels, bands, gdal.GDT_Float64)

    geotrans = example.GetGeoTransform()  # get GeoTranform from existed 'data0'
    proj = example.GetProjection()  # you can get from a exsited tif or import
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    for b in range(bands):
        dataset.GetRasterBand(b + 1).WriteArray(array[b, :, :])

    dataset.FlushCache()
    return dataset


ccdcFnAfter = '/home/centos/srlite/LR/pitkus-point-demo/GEE-CCDC-pitkusPointSubset-esri102001-30m-after-edit.tif'
ccdcFn = '/home/centos/srlite/LR/pitkus-point-demo/GEE-CCDC-pitkusPointSubset-esri102001-30m-after-edit.tif'
evhrFn = '/home/centos/srlite/LR/pitkus-point-demo/GDAL-EVHR-WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPointSubset-esri102001-30m-avg-cog.tif'

# TODO - use gdal_edit or equivalent to dynamically convert GEE CCDC output to proper projection ESRI:102001

# CCDC - Create 3-D Numpy array from original dataset - shape = (4, 621, 806)
ccdcFnNdArr = gdal_array.LoadFile(ccdcFnAfter)
# Replace 'nan' values with -9999
ccdcFnNdArrNoNan = np.nan_to_num(ccdcFnNdArr, copy=True, nan=-9999)
# Convert Numpy array to GDAL dataset for memwarp call
ccdcFnNdArrNoNanDs = save_rasters_memory(ccdcFnNdArrNoNan, ccdcFnAfter)

# EVHR - Open file as GDAL dataset directly since on 'nan' values are contained
evhrDs = gdal.Open(evhrFn, gdal.GA_ReadOnly)

#  Call memwarp to get the 'intersection' of the CCDC & EVHR datasets
ds_list = warplib.memwarp_multi(
    [ccdcFnNdArrNoNanDs, evhrDs], res='max',
    extent='intersection', t_srs='first', r='average')

# Name the datesets and verify raster count
ccdcDsAfterMemwarp = ds_list[0]
evhrDsAfterMemwarp = ds_list[1]
numBands = ccdcDsAfterMemwarp.RasterCount

# Get Numpy array from dataset for linear regression (per band)
for b in range(numBands):
    ccdcBand = ccdcDsAfterMemwarp.GetRasterBand(b + 1).ReadAsArray()
    evhrBand = evhrDsAfterMemwarp.GetRasterBand(b + 1).ReadAsArray()

    lr = SimpleLinearRegression(ccdcBand, evhrBand)
    lr.run()

exit();
# -------- THE MAGIC COMBINATION --------
#dataset = gdal.Open(r'/att/nobackup/gtamkin/srlite/LR/2-fairbanks-august/WV02_20180811_M1BS_1030010080D1AB00-toa.tif')
#src0 = rasterio.open('/att/nobackup/gtamkin/srlite/LR/1-fairbanks-october/WV02_20181005_M1BS_1030010084131B00-toa.tif')
#src1 = rasterio.open('/att/nobackup/gtamkin/srlite/LR/1-fairbanks-october/WV02_20181005_M1BS_1030010084131B00-toa_30m.tif')


#########  EPSG-4326 experiment - pitkus-point-demo with 30m generated from GDAL ##########
#src2 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GEE-CCDC-PitkusPointSubset-epsg4326-30m.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GDAL-WV02_20110818_M1BS_103001000CCC9000-toa-PitkusPointSubset-30m-avg-cog.tif')

#########  EPSRI:102001 experiment - pitkus-point-demo with 30m generated from GEE and corrected using gdal_edit.py  ##########
src2 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GEE-CCDC-pitkusPointSubset-esri102001-30m-after-edit.tif')
##src2 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GEE-CCDC-pitkusPointSubset-esri102001-30m-before-edit.tif')
src3 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GDAL-EVHR-WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPointSubset-esri102001-30m-avg-cog.tif')

#########  EPSG-4326 experiment - pitkus-point-demo with 30m generated from GEE ##########
#src2 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GEE-CCDC-PitkusPointSubset-epsg4326-30m.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/pitkus-point-demo/GEE-WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPointSubset-epsg4326-30m.tif')

#########  EPSG-4326 experiment - Pitkus Point Subset##########
#src2 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/CCDC-PitkusPointSubset-epsg4326.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/YKD-EVHR-PitkusPointSubset-epsg4326.tif')

#########  EPSG-102001 experiment - Pitkus Point Subset##########
#src2 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/CCDC-PitkusPointSubset-epsg102001.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/YKD-EVHR-PitkusPointSubset-epsg102001.tif')

#########  EPSG-4326 experiment - Pitkus Point ##########
#src2 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/CCDC-PitkusPoint-epsg4326.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/YKD-EVHR-PitkusPoint-epsg4326.tif')

#########  ESRI-102001 experiment - farmers loop ##########
#src2 = rasterio.open('/home/centos/srlite/LR/1-fairbanks-october/WV02_1030010084131B00-toa_30m-ESRI-102001.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/1-fairbanks-october/CCDC-2018-10-05-reprojected-clipped-farmersloop-all-epsg-102001-scale30.tif')

#########  data after clipping to eliminate weird rectangular image ##########
#src2 = rasterio.open('/home/centos/srlite/LR/1-fairbanks-october/LR-20210909/WV-2018-10-05-clipped-farmersloop-b4-epsg-32606-scale30.tif')
#src3 = rasterio.open('/home/centos/srlite/LR/1-fairbanks-october/LR-20210909/CCDC-2018-10-05-reprojected-clipped-farmersloop-all-epsg-32606-scale30.tif')

########  Pitkus Point 102001##########
##src2 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/YKD-PitkusPoint-epsg-102001-scale30_20110818_WV02_013285751010.tif')
##src3 = rasterio.open('/home/centos/srlite/LR/3-pitkus-point-20110818/CCDC-PitkusPoint-epsg-102001-scale30_20110818.tif')

#########  data upon kickoff with Paul ##########
##src2 = rasterio.open('/home/centos/srlite/LR/1-fairbanks-october/WV-2018-10-05-clipped-b4-epsg-32606.tif')
##src3 = rasterio.open('/home/centos/srlite/LR/1-fairbanks-october/CCDC-2018-10-05-reprojected-clipped-nir-epsg-32606.tif')
#src4 = rasterio.open('/att/nobackup/gtamkin/srlite/LR/2-fairbanks-august/WV02_20180811_M1BS_1030010080D1AB00-toa.tif')

#print(src0, src0.name, src0.mode, src0.closed)
#print(src1, src1.name, src1.mode, src1.closed)
print(src2, src2.name, src2.mode, src2.closed)
print(src3, src3.name, src3.mode, src3.closed)
#print(src4, src4.name, src4.mode, src4.closed)

# since there are 4 bands
# we store in 4 different variables
bands = [src2,src3]
for band in bands:
   print(band.shape)
   print(band.descriptions)
   print(band.profile)
   print(band.transform)
   print(band.crs)

for index in range(1,5):

    # read in bands from image
    wvB4Band = src3.read(index)
    ccdcNirBand = src2.read(index)

    lr = SimpleLinearRegression(ccdcNirBand, wvB4Band)
#    linear = SimpleLinearRegression(wvB4Band, ccdcNirBand)
    lr.run()

exit()

#wvB4Band = src2.read(4)
#ccdcNirBand = src3.read(4)

#r_band0 = src0.read()
#r_band1 = src1.read()
#r_band2 = src2.read()
#r_band3 = src3.read()
#r_band4 = src4.read()
#r_band1 = src1.read(1)
#r_band2 = src2.read(1)
#r_band3 = src2.read(1)
#print(r_band0.shape, r_band1.shape, r_band2.shape, r_band3.shape, r_band4.shape)

print()
print(ccdcNirBand)

#linear = SimpleLinearRegression(ccdcNirBand, wvB4Band)
lr = SimpleLinearRegression(wvB4Band, ccdcNirBand)
lr.run()
exit()

pyplot.imshow(r_band0[0], cmap='Reds')
pyplot.show()

pyplot.imshow(r_band1[0], cmap='pink')
pyplot.show()

pyplot.imshow(r_band2[0], cmap='Blues')
pyplot.show()

pyplot.imshow(r_band3[0], cmap='Greens')
pyplot.show()

pyplot.imshow(r_band4[0], cmap='bone_r')
pyplot.show()

src = src0
for i, dtype, nodataval in zip(src.indexes, src.dtypes, src.nodatavals):
    print(i, dtype, nodataval)
src = src1
for i, dtype, nodataval in zip(src.indexes, src.dtypes, src.nodatavals):
    print(i, dtype, nodataval)
src = src2
for i, dtype, nodataval in zip(src.indexes, src.dtypes, src.nodatavals):
    print(i, dtype, nodataval)
src = src3
for i, dtype, nodataval in zip(src.indexes, src.dtypes, src.nodatavals):
    print(i, dtype, nodataval)
src = src4
for i, dtype, nodataval in zip(src.indexes, src.dtypes, src.nodatavals):
    print(i, dtype, nodataval)

exit()
