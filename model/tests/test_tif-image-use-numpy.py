
# Step 1: Import the modules and open the file.,
from osgeo import gdal
import matplotlib.pyplot as plt
from lr.SimpleLinearRegression import SimpleLinearRegression

#dataset = gdal.Open(r'/att/nobackup/gtamkin/srlite/LR/2-fairbanks-august/WV02_20180811_M1BS_1030010080D1AB00-toa.tif')
#dataset = gdal.Open(r'/att/nobackup/gtamkin/srlite/LR/1-fairbanks-october/CCDC-2018-10-05-reprojected-clipped-nir-epsg-32606.tif')
#dataset = gdal.Open(r'WV-2018-10-05-clipped-b4-epsg-32606.tif')

#src1 = rasterio.open('/att/nobackup/gtamkin/srlite/LR/1-fairbanks-october/WV02_20181005_M1BS_1030010084131B00-toa_30m.tif')
#src2 = rasterio.open('/att/nobackup/gtamkin/srlite/LR/1-fairbanks-october/WV-2018-10-05-clipped-b4-epsg-32606.tif')
#src3 = rasterio.open('/att/nobackup/gtamkin/srlite/LR/1-fairbanks-october/CCDC-2018-10-05-reprojected-clipped-nir-epsg-32606.tif')

#########  data after clipping to eliminate weird rectangluar image ##########
ccdcClippedDataset = \
   gdal.Open(r'/home/centos/srlite/LR/1-fairbanks-october/LR-20210909/CCDC-2018-10-05-reprojected-clipped-fairbanks-nir-epsg-32606-scale30.tif')
wv1ClippedDataset = \
   gdal.Open(r'/home/centos/srlite/LR/1-fairbanks-october/LR-20210909/WV-2018-10-05-clipped-fairbanks-b4-epsg-32606-scale30.tif')
wv1Dataset = \
   gdal.Open(r'/home/centos/srlite/LR/1-fairbanks-october/WV02_20181005_M1BS_1030010084131B00-toa_30m.tif')

#########  data upon kickoff with Paul ##########
#ccdcClippedDataset = gdal.Open(r'/home/centos/srlite/LR/1-fairbanks-october/CCDC-2018-10-05-reprojected-clipped-nir-epsg-32606.tif')
#wv1ClippedDataset = gdal.Open(r'/home/centos/srlite/LR/1-fairbanks-october/WV-2018-10-05-clipped-b4-epsg-32606.tif')
#wv1Dataset = gdal.Open(r'/home/centos/srlite/LR/1-fairbanks-october/WV02_20181005_M1BS_1030010084131B00-toa_30m.tif')

#Step 2: Count the number of bands.,
bands = [wv1Dataset,wv1ClippedDataset,ccdcClippedDataset]
for band in bands:
   print(band.RasterCount)
#print(dataset.RasterCount)
#Step 3: Fetch the bands,,

#To fetch the bands we use GDAL’s GetRasterBand(int). ,
#Note that the value of int we pass will always start from 1 (indexing of bands starts from 1), band1 = dataset.GetRasterBand(1) # Red channel,
# since there are 4 bands,
# we store in 4 different variables,
ccdcClippedDatasetNirBandRaster = ccdcClippedDataset.GetRasterBand(4) # NIR channel,
wv1ClippedDatasetB4BandRaster = wv1ClippedDataset.GetRasterBand(1) # B4 channel,
wv1DatasetB4BandRaster = wv1Dataset.GetRasterBand(4) # B4 channel,
#band4 = dataset.GetRasterBand(4) # NIR channel
bandArrayRasters = [wv1DatasetB4BandRaster,wv1ClippedDatasetB4BandRaster,ccdcClippedDatasetNirBandRaster]
for band in bandArrayRasters:
   print(band)

# GDAL provides ReadAsArray() method that converts the bands into numpy arrays and returns them. ,
ccdcClippedDatasetNirBandRasterArray = ccdcClippedDatasetNirBandRaster.ReadAsArray()
wv1ClippedDatasetB4BandRasterArray = wv1ClippedDatasetB4BandRaster.ReadAsArray()
wv1DatasetB4BandRasterArray = wv1DatasetB4BandRaster.ReadAsArray()

ccdcClippedDatasetNirBandRasterArrayFlat = ccdcClippedDatasetNirBandRasterArray.flatten()
wv1ClippedDatasetB4BandRasterArrayFlat = wv1ClippedDatasetB4BandRasterArray.flatten()
wv1DatasetB4BandRasterArrayFlat = wv1DatasetB4BandRasterArray.flatten()
bandArrays = [wv1DatasetB4BandRasterArray,wv1ClippedDatasetB4BandRasterArray,ccdcClippedDatasetNirBandRasterArray]
for band in bandArrays:
   print(band)
bandArrays = [wv1DatasetB4BandRasterArrayFlat,wv1ClippedDatasetB4BandRasterArrayFlat,ccdcClippedDatasetNirBandRasterArrayFlat]
for band in bandArrays:
   print(band)

#Step 5: Plotting the arrays using imshow().,
import numpy as np

#img = np.dstack((wv1ClippedDatasetB4BandRasterArrayFlat[0], ccdcClippedDatasetNirBandRasterArrayFlat[0]))
#img = np.dstack((wv1ClippedDatasetB4BandRasterArray[0], ccdcClippedDatasetNirBandRasterArray[0]))
#img = np.dstack((b1, b2, b3))

#f = plt.figure()
#plt.imshow(img)
#plt.savefig('Tiff.png')
#plt.show()

#wv1DatasetB4BandRasterArrayFlat
#linear = SimpleLinearRegression(wv1ClippedDatasetB4BandRasterArrayFlat, ccdcClippedDatasetNirBandRasterArrayFlat)
lr = SimpleLinearRegression(wv1ClippedDatasetB4BandRasterArrayFlat, ccdcClippedDatasetNirBandRasterArray)
lr.run()

exit()
############
from PIL import Image
img = Image.open('/att/nobackup/gtamkin/srlite/LR/2-fairbanks-august/WV02_20180811_M1BS_1030010080D1AB00-toa.tif'),
img.show()
import skimage.io as skio
import scipy
from PIL import Image

file_path=('/att/nobackup/gtamkin/srlite/LR/2-fairbanks-august/WV02_20180811_M1BS_1030010080D1AB00-toa.tif'),
#print("\n\The selected stack is a .tif:\n\"),
dataset = Image.open(file_path),
h,w = np.shape(dataset),
tiffarray = np.zeros((h,w,dataset.n_frames))
for i in range(dataset.n_frames):
   dataset.seek(i),
   tiffarray[:,:,i] = np.array(dataset)
expim = tiffarray.astype(np.double);
print(expim.shape)
#Visualizing Tiff File Using Matplotlib and GDAL using Python,
# https://www.geeksforgeeks.org/visualizing-tiff-file-using-matplotlib-and-gdal-using-python/
