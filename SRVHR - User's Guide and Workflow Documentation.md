# $SR_{VHR}$ v.2.0.0 User’s Guide & Workflow Documentation

$SR_{VHR}$ (previously referred to as *SR-lite*): surface reflectance estimates for very high resolution multispectral imagery 

## Overview: What is $SR_{VHR}$?

$SR_{VHR}$ [1] refers to an algorithm to empirically estimate surface reflectance (SR) from very high resolution multispectral (VHR) imagery (eg, Maxar). The $SR_{VHR}$ workflow (Figure 1\) is formalized in containerized code that returns SR estimates for VHR imagery (including WorldView-2/3/4 and GeoEye-1) at 2 m spatial resolution from input top-of-atmosphere (TOA) reflectance estimates and spatially and temporally coincident reference synthetic Landsat SR. Results are derived per-band using a linear model where $SR_{VHR} \= m(TOA_{VHR}) \+ b$, and returned as cloud optimized GeoTIFFs. Version 2 of $SR_{VHR}$ uses the continuous change detection and classification (CCDC) algorithm [2] to generate modeled synthetic reference SR for the day corresponding to the input VHR dataset.

![SRvhr_workflow](https://github.com/user-attachments/assets/43d79f02-ee59-4616-b3a6-e3917d2afd14)

Figure 1\. The $SR_{VHR}$ workflow models land surface reflectance using cloud-masked input top-of-atmosphere VHR reflectance and coincident reference surface reflectance estimates from synthetic Landsat. This workflow is run in containerized code to produce 2 m cloud-optimized GeoTIFFs.

## Quick reference for running $SR_{VHR}$ 

$SR_{VHR}$ runs in a container, on a per-TOA basis, and requires 3 inputs:

1. $TOA_{VHR}$ geotiff  
2. $TOA_{VHR}$ cloudmask geotiff  
3. Reference surface reflectance multispectral geotiff

The $SR_{VHR}$ *Python* application is called using a *Singularity* container from a Linux terminal with the following general format:

| * singularity run \-B \<local path(s) to mount\> \<container name\> python \<python application\> \<runtime parameters\>* |
| :---- |

### Mandatory runtime parameters:

* \-toa\_dir \- directory containing $TOA_{VHR}$ 2m (default suffix \= `toa.tif`).  Note that a single, fully-qualified path to a single $TOA_{VHR}$ can be provided here instead of a directory.  
* \-target\_dir \- directory container model data (default suffix \= `ccdc.tif`)  
* \-cloudmask\_dir \- directory containing cloudmasks (default suffix \= `toa.cloudmask.v1.2.tif`)  
* \-output\_dir \- directory containing results for this specific invocation 

### Optional runtime parameters:

* \-bandpairs \- list of band pairs to be processed \[(model band name B, TOA band name B), (model band name R, TOA band name R) …\] *\[default="\[\['blue\_ccdc', 'BAND-B'\], \['green\_ccdc', 'BAND-G'\], \['red\_ccdc', 'BAND-R'\],\['nir\_ccdc', 'BAND-N'\],\['blue\_ccdc', 'BAND-C'\], \['green\_ccdc', 'BAND-Y'\], \['red\_ccdc', 'BAND-RE'\], \['nir\_ccdc', 'BAND-N2'\]\]"\]*  
* \--regressor \- choose regression algorithm \[‘rma’,'simple', 'robust'\] *\[default \= ‘rma’\]*  
* \--cloudmask \- apply cloud mask values to common mask *\[default \= False\]*  
* \--csv \- generate comma-separated values (CSV) for output statistics *\[default \= True\]*  
* –csv-dir \- directory path to receive statistics files *\[default \= output\_dir, “None” turns off –csv flag\]*  
* –err-dir \- directory path to receive error files *\[default \= output\_dir\]*  
* –warp-dir \- directory path to receive interim warped files *\[default \= output\_dir\]*  
* \--band8 \- create simulated bands for missing CCDC bands (C/Y/RE/N2) *\[default \= False\]*  
* \--xres \- specify target X resolution *(default \= 30.0)*  
* \--yres \- specify target Y resolution *(default \= 30.0)*  
* \--toa\_suffix \- specify TOA file suffix *(default \= `toa.tif`')*  
* \--target\_suffix \- specify TARGET file suffix *(default \= `ccdc.tif`')*  
* \--cloudmask\_suffix \- specify CLOUDMASK file suffix *(default \= `toa.cloudmask.v1.2.tif`')*  
* \--clean \- overwrite previous output *(default \= True)*  
* \--thmask \- apply threshold mask values to common mask *(default \= False)*  
* \--thrange \- choose quality flag values to mask \[*default='-100, 2000'\]*  
* \--pmask \- suppress negative values from common mask *\[default \= False\]*  
* \--debug \- diagnostic level *\[0=None, 1=trace\]*  
* \--batch \- name of run *\[default \= None\]*  
* \--log \- create log file for stdout/stderr *\[default \= False\]*

Sample Invocation:

| *$ singularity run \-B /panfs/ccds02/nobackup/people/iluser/projects/srlite,/panfs/ccds02/nobackup/people/gtamkin,/home/gtamkin/.conda/envs,/run,/explore/nobackup/people/gtamkin/dev,/explore/nobackup/projects/ilab/data/srlite/products /explore/nobackup/people/iluser/ilab\_containers/srlite\_1.0.1.sif python /usr/local/ilab/srlite/srlite/view/SrliteWorkflowCommandLineView.py \-toa\_dir /panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline \-target\_dir /panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline \-cloudmask\_dir /panfs/ccds02/nobackup/people/iluser/projects/srlite/test/input/baseline \-bandpairs "\[\['blue\_ccdc', 'BAND-B'\], \['green\_ccdc', 'BAND-G'\], \['red\_ccdc', 'BAND-R'\], \['nir\_ccdc', 'BAND-N'\], \['blue\_ccdc', 'BAND-C'\], \['green\_ccdc', 'BAND-Y'\], \['red\_ccdc', 'BAND-RE'\], \['nir\_ccdc', 'BAND-N2'\]\]"  \-output\_dir /explore/nobackup/projects/ilab/data/srlite/products/srlite\_1.0.0-baseline/srlite\_1.0.0-ADAPT-cli/Whitesands/srlite-1.0.0-rma-baseline  \--regressor rma \--debug 1 \--pmask \--cloudmask \--clean \--csv \--band8*  |
| :---- |

The output data is delivered to a directory specified in the program call (\-output\_dir). There are two outputs:

1. The image data is output in COG format with the following naming convention:  `\<SENSOR\>\_\<YYYYMMDD\>\_\<CATID\>-sr-02m.tif`.   
2. The regression results are output as *.csv* files (see Table 1). They contain linear model (slope and intercept) coefficients along with $SR_{VHR}$ performance statistics.  .

## Methodology

The $SR_{VHR}$ workflow consists of the methodology detailed below. In summary, it stacks all input layers into a common modeling grid (eg, 30 m; coarser than the input $TOA_{VHR}$) grid for the input $TOA_{VHR}$ spatial extent, derives a mask from invalid data in each of the 3 input datasets, applies that mask to the input TOA and reference SR to remove all invalid pixel, builds an SR model at 30 m resolution, and applies that model to the original $TOA_{VHR}$ at 2m spatial resolution.

### Computing $TOA_{VHR}$ reflectance for VHR

The Enhanced Very High Resolution Workflow (EVHR) [3] is used to produce $TOA_{VHR}$ geotiffs for any VHR scene or a mosaic of a sequential collection of scenes (strip). A multi-spectral stacked geotiff is returned that is georeferenced to the local Universal Transverse Mercator coordinate system in a grid with a resolution native to the input $TOA_{VHR}$ from the VHR imagery (2 m).

### Identifying valid pixels by masking cloud cover in $TOA_{VHR}$

To identify valid surface pixels, we applied a convolutional neural net (CNN) algorithm to mask cloudcover for each Worldview VHR dataset [4]. The mask is returned as a binary map that separates cloud from non-cloud pixels. The development of this algorithm is on-going. In some cases, transparent cirrus clouds are not identified as such, while some very bright non-cloud surfaces (smooth snow cover or bright lichen ground cover extents) may be mis-classified as clouds.

### Compiling reference surface reflectance

To compile a reference of surface reflectance ($SR_{reference}$) we derived Landsat-derived SR estimates using the CCDC algorithm. CCDC model parameters are generated from all available Landsat 4/5/7/8/9 Tier 1 Level 2 surface reflectance observations (masked to exclude cloud, cloud shadows, snow, pixels that are saturated in any band, and gaps). Then, we use the projection, bounding box and acquisition date of the input $TOA_{VHR}$ scene to generate a synthetic (modeled) map of estimated SR based on the CCDC model parameters. The CCDC parameters are based on the Landsat time-series inputs and incorporate seasonality,trends, and disturbances in the Landsat surface reflectance record.

### Constructing a modeling data stack: re-gridding and masking

We compile an input filename list of the 3 datasets ($SR_{reference}$, input $TOA_{VHR}$, and cloud cover mask of input $TOA_{VHR}$) that will be warped, aggregated, and masked before model building.

#### Regridding

The input files are regridded by warping each to the projection of the input $TOA_{VHR}$ and re-gridding to a coarsened (30 m) pixel resolution. We warp the reference data to match the projection of the input $TOA_{VHR}$ and re-grid using mean for the reflectance bands and mode for the cloud mask. Regridding the 2 m inputs to the coarser spatial resolution of the 30 m reference surface reflectance at this stage makes for more efficient model building. 

#### Masking

After regridding, we build a common mask that includes all “no data” collected from both reference and input $TOA_{VHR}$ datasets. This mask is thus a union of all input “nodata”, and includes as “nodata” the masked cloudcover pixels from the input $TOA_{VHR}$.  Both reference and input $TOA_{VHR}$ are masked with this common mask so that the same set of pixels are present in each dataset. At this point, each input $TOA_{VHR}$ pixel has a corresponding reference pixel, and the data is ready for model building. 

### Building and applying the $SR_{VHR}$ model with the data stack

We use the re-gridded and mask data stack to build a linear model to describe the relationship of the input $TOA_{VHR}$ (dependent) to the $SR_{reference}$ (independent). The models are applied bandwise for the blue, green, red, and near-infrared bands (Band 7, or NIR1, in the case of Worldview-2/3/4), where a given $TOA_{VHR}$ band is matched to the closest corresponding reference band based on the central wavelength. We provide a choice of 3 linear models that are applied to each bandwise pairing (eg. $TOA_{VHR_blue}$ \~ f($SR_{reference_blue}$) of dependent and independent data in this 30 m data stack. These choices are:

1. *simple:*		ordinary least squares regression from *sklearn* using the *LinearRegressor* module.  
2. *rma* : 		Python’s *pylr2* implementation of reduced major axis (RMA) regression to each  
3. *robust*: 	Python’s *sklearn* implementation using the *HuberRegressor* module.

The bandwise model fit from that is returned based on the model choice is then applied back to corresponding input $TOA_{VHR}$ band (2 m). This is done for each band of the input $TOA_{VHR}$ to return the multi-band $SR_{VHR}$ surface reflectance estimates at 2 m spatial resolution. For the special case of an 8-band input $TOA_{VHR}$, the model coefficients for the 4 extra bands (that are not present in the reference Landsat-derived SR input) are weighted coefficients derived from (in the case where the focal input $TOA_{VHR}$ band falls between 2 reference bands) the two nearest bands (yellow and red-edge), or the nearest band (coastal blue and NIR2).

## Description of Output

Output images are returned as multi-band Cloud Optimized GeoTIFFs (COGs) with a corresponding linear model result table (CSVs). These table includes:

1.  a summary of the correction model coefficients applied bandwise to each pixel of the $TOA_{VHR}$,   
2. statistical results of the $SR_{VHR}$ comparison with input $SR_{reference}$ data and input $TOA_{VHR}$  
   

The image products inherit the projection, extent, and cell sizeof the EVHR $TOA_{VHR}$ outputs, typically the local UTM coordinate system

Table 1\. An example of the output CSV table for each $SR_{VHR}$ result. This table contains correction model coefficients, which result from the linear regression between 30m $TOA_{VHR}$ and the chosen $SR_{reference}$ (e.g., synthetic Landsat) 30m, that are then applied to the input 2m $TOA_{VHR}$ to create the output grid (COG format).  The performance statistics compare $SR_{VHR}$ outputs to corresponding reference values. Note: in this example output there are no performance statistics for input $TOA_{VHR}$ bands that don’t have corresponding reference bands (eg. RedEdge is present in Worldview 2 input $TOA_{VHR}$, but $SR_{reference}$ does not have a corresponding band). Therefore, the output coefficients are the result of the weights used to calculate a correction from the nearest existing reference band.

![][image2]

*Table 2\. Description of output linear model results table column names.*

| *Column Name* | *Description* |
| :---- | :---- |
| *band\_names:* | *the name of the input TOA bands* |
| *model* | *he name of the linear model choice* |
| *intercept, slope* | *the values of the coefficients for each model relating input TOA to reference SR* |
| *r2\_score* | *coefficient of determination regression score, representing the proportion of variance of the dependent variable explained by the independent variables. https://scikit-learn.org/stable/modules/model\_evaluation.html\#r2-score* |
| *explained\_variance* | *the “r-squared value” representing the proportion of variance between the dependent and independent variables explained by the linear model. Similar to r2\_score but does not account for systematic offsets in the prediction.* |
| *mae* | *mean absolute error in the prediction*  |
| *mbe* | *mean bias in the prediction* |
| *mape* | *mean absolute percentage error from the model residuals*  |
| *medea* | *median absolute error (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median\_absolute\_error.html\#sklearn.metrics.median\_absolute\_error)* |
| *mse* | *mean squared error from the model residuals (https://scikit-learn.org/stable/modules/model\_evaluation.html\#mean-squared-error)* |
| *rmse* | *root mean squared error from the model residuals (y)* |
| *mean\_ccdc\_sr* | *mean surface reflectance of the reflectance target (CCDC) in the final (fully masked) model array.* |
| *mean\_evhr\_srlite* | *mean surface reflectance of the $SR_{VHR}$ product in the final (fully masked) model array.* |
| *mae\_norm* | *normalized mean absolute error, normalized by dividing the mae by the mean CCDC reflectance.*  |
| *rmse\_norm* | *normalized root mean square error, normalized by dividing the rmse by the mean CCDC reflectance.* |

## Evaluating results

$SR_{VHR}$ results can be evaluated in 2 ways, on an individual image basis, and by batch for a set of many images for a particular area.

### Individual image evaluation

For each individual $SR_{VHR}$ image output we compare for each MS band:

1. the image’s $SR_{VHR}$  vs. input $TOA_{VHR}$ reflectance values (30 m).  
2. the image’s $SR_{VHR}$ vs. input reference reflectance values (30 m).

A suite of regression metrics for describing the strength of the model used to derive the output $SR_{VHR}$ from the input $TOA_{VHR}$ are returned to a CSV file.

### Global evaluation

For a batch of $SR_{VHR}$ image output, typically consisting of imagery with a variety of sun-sensor-geometry image acquisition characteristics, we compare model results of the image’s $SR_{VHR}$ vs $TOA_{VHR}$ reflectance values. Plots showing all individual linear model results show which images were likely not well corrected (slopes significantly less than 1).  This can often be a way for users to easily identify images of particularly poor quality relative to the rest of their data. 

## Future considerations

$SR_{VHR}$ is not yet designed to handle the following circumstances:

1. **Snow**: Snow is masked from CCDC so reference image during partial snow season generally will depict snow-free reflectance. We should be masking snow from EVHR before applying any regression with CCDC reference, however we do not currently have a snow mask to use.  
2. **Cloud shadow**: Artificially dark areas in EVHR that are not masked can greatly affect regression  
3. **Water**: Especially with high sediment loads or ice, not a stable reflectance target  
4. **Saturation**: More common in QB2/GE1 images. Often associated with snow. Regression not appropriate with saturated pixels. Saturated pixels are excluded from the CCDC model.  
5. **Out of bounds values**: Reflectance \< 0 or \> 1 are sometimes far outside the bound of the valid 0-1 range.

$SR_{VHR}$ output in an upcoming version will include:

1. VHR acquisition geometry plotted on polar coordinates, useful for understanding some of the variation within a batch of $SR_{VHR}$ output.  
2. $SR_{VHR}$ output assessment notebooks.

$SR_{VHR}$ adjustments to consider:

1. Geographic shift: CCDC (generated on a 30m grid) gets an extra arbitrary resampling step to match one corner of the $TOA_{VHR}$ image. This results in a subpixel shift, possibly accompanied by a smoothing if NN resampling is not used.  
2. Algorithm speed: mode resampling for cloud mask could be switched to average as long as pixels with a decimal value are then classified as cloud (conservative \- more cloud area than the input). Mode may be an order of magnitude or more slower and is one of the slowest steps in the algorithm.

## References

[1]	P. M. Montesano *et al.*, “Surface Reflectance From Commercial Very High Resolution Multispectral Imagery Estimated Empirically With Synthetic Landsat (2023),” *IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.*, pp. 1–10, 2024, doi: 10.1109/JSTARS.2024.3456587.]  
[2]	Z. Zhu, C. E. Woodcock, C. Holden, and Z. Yang, “Generating synthetic Landsat images based on all available Landsat data: Predicting Landsat surface reflectance at any given time,” *Remote Sens. Environ.*, vol. 162, pp. 67–83, Jun. 2015, doi: 10.1016/j.rse.2015.02.009.]  
[3]	C. S. R. Neigh *et al.*, “An API for Spaceborne Sub-Meter Resolution Products for Earth Science,” *Int. Geosci. Remote Sens. Symp. IGARSS*, pp. 5397–5400, Jul. 2019, doi: 10.1109/IGARSS.2019.8898358.]  
[4]	J. A. Caraballo-Vega *et al.*, “Optimizing WorldView-2, \-3 cloud masking using machine learning approaches,” *Remote Sens. Environ.*, vol. 284, p. 113332, Jan. 2023, doi: 10.1016/j.rse.2022.113332.]  
