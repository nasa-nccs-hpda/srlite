'''
Example of preparing 2 image inputs for comparitive analysis

    Viewing input projections
    warping input arrays to common grid
    getting the common masked data (the union of each mask)
    applying the common mask to the warped arrays
'''
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np

from pygeotools.lib import iolib, malib, geolib, filtlib, warplib

import osgeo
from osgeo import gdal, ogr, osr
print(osgeo.gdal.VersionInfo())

from core.model.SystemCommand import SystemCommand

#maindir = '/att/nobackup/gtamkin/srlite/pitkus-point-demo'
#r_fn_evhr = os.path.join(maindir, 'EVHR-WV02_20110818_M1BS_103001000CCC9000-toa.tif')
#r_fn_ccdc = os.path.join(maindir, 'GEE-CCDC-pitkusPointSubset-esri102001-30m.tif')
#r_fn_ccdc = os.path.join(maindir, 'GEE-CCDC-PitkusPointSubset-epsg4326-30m.tif')

#maindir = '/att/nobackup/pmontesa/userfs02/projects/srlite/12-21-2021'

r_fn_ccdc = '/home/centos/srlite/srlite-workflow-01032022/srlite/model/datasets/ccdc/ccdc_20110818.tif'
r_fn_evhr = '/home/centos/srlite/srlite-workflow-01032022/srlite/model/datasets/wv/WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif'

#r_fn_ccdc = os.path.join(maindir, 'ccdc_20110818.tif')
#r_fn_evhr = os.path.join(maindir, 'WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif')

fn_list = [r_fn_ccdc, r_fn_evhr]

OUTDIR = '/home/centos/srlite/srlite-workflow-01032022/results'

#OUTDIR = '/att/nobackup/pmontesa/userfs02/projects/srlite/misc'

#!source activate ilab-kernel; python /att/gpfsfs/home/pmontesa/code/evhr/utm_proj_select.py ${r_fn_hrsi}

# This wont run here until osgeo import gets sorted out
#os.system('python /att/gpfsfs/home/pmontesa/code/evhr/utm_proj_select.py '+ os.path.join(maindir,r_fn_hrsi))

#Look at the CCDC projection

print(r_fn_ccdc)
ccdc_ds = iolib.fn_getds(r_fn_ccdc)
print(ccdc_ds.GetProjection())
#!gdalinfo /att/nobackup/pmontesa/userfs02/projects/srlite/12-21-2021/ccdc_20110818.tif
command = 'gdalinfo ' + r_fn_ccdc
print(SystemCommand(command))

#Look at the EVHR projection

evhr_ds = iolib.fn_getds(r_fn_evhr)
print(evhr_ds.GetProjection())
#!gdalinfo '/att/gpfsfs/briskfs01/ppl/pmontesa/userfs02/projects/srlite/12-21-2021/WV02_20110818_M1BS_103001000CCC9000-toa-pitkusPoint-cog.tif'
command = 'gdalinfo ' + r_fn_evhr
print(SystemCommand(command))

#INPUT ARRAYS

#maps

ma_list = [iolib.fn_getma(fn) for fn in fn_list]

figsize = (10, 5)

fig, axa = plt.subplots(nrows=1, ncols=len(fn_list), figsize=figsize, sharex=False, sharey=False)
for i, ma in enumerate(ma_list):
    f_name = fn_list[i]

    divider = make_axes_locatable(axa[i])
    cax = divider.append_axes('right', size='2.5%', pad=0.05)
    im1 = axa[i].imshow(ma, cmap='RdYlGn', clim=malib.calcperc(ma, perc=(10, 90)))
    cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend='max')
    axa[i].set_title(os.path.split(f_name)[1], fontsize=10)
    cb.set_label('Reflectance (%)')

plt.tight_layout()

#INPUT ARRAYS

#histograms

#ma_list = [iolib.fn_getma(fn) for fn in fn_list]

figsize = (10, 3)
fig, axa = plt.subplots(nrows=1, ncols=len(ma_list), figsize=figsize, sharex=True, sharey=True)

for i, ma in enumerate(ma_list):
    f_name = fn_list[i]
    print(f" {ma.count()} valid pixels in INPUT MASKED ARRAY version of {f_name}")

    h = axa[i].hist(ma.compressed(), bins=512, alpha=0.75)
    axa[i].set_title(os.path.split(f_name)[1], fontsize=10)

plt.tight_layout()

#To intersect these images, use warplib

warp_ds_list = warplib.memwarp_multi_fn(fn_list, res='first', extent='intersection', t_srs='first', r='near')
warp_ma_list = [iolib.ds_getma(ds) for ds in warp_ds_list]

ccdc_warp_ma, evhr_warp_ma = warp_ma_list
#evhr_warp_ds.GetProjection()
#evhr_warp_ma = iolib.ds_getma(evhr_warp_ds)
#evhr_warp_ma = np.where(hrsi_warp_ma >= 0)
print('\n',
    ccdc_warp_ma.shape,
    evhr_warp_ma.shape
)

figsize = (10, 5)

fig, axa = plt.subplots(nrows=1, ncols=len(fn_list), figsize=figsize, sharex=False, sharey=False)
for i, ma in enumerate(warp_ma_list):
    f_name = fn_list[i]

    divider = make_axes_locatable(axa[i])
    cax = divider.append_axes('right', size='2.5%', pad=0.05)
    im1 = axa[i].imshow(ma, cmap='RdYlGn', clim=malib.calcperc(ma, perc=(1, 95)))
    cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend='max')
    axa[i].set_title(os.path.split(f_name)[1], fontsize=10)
    cb.set_label('Reflectance (%)')

plt.tight_layout()

#WARPED MASKED ARRAY

#maps

warp_ma_list = [ccdc_warp_ma, evhr_warp_ma]

figsize = (10, 3)
fig, axa = plt.subplots(nrows=1, ncols=len(warp_ma_list), figsize=figsize, sharex=True, sharey=True)

#WARPED MASKED ARRAY

#histograms

for i, ma in enumerate(warp_ma_list):
    f_name = os.path.split(fn_list[i])[1]
    print(f" {ma.count()} valid pixels in WARPED MASKED ARRAY version of {f_name}")

    h = axa[i].hist(ma.compressed(), bins=512, alpha=0.75)
    axa[i].set_title(f_name, fontsize=10)

plt.tight_layout()

print("Strutter")

