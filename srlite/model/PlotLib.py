#!/usr/bin/env python
# coding: utf-8
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from plotnine import ggplot, aes, geom_smooth, geom_bin2d, geom_abline
from pygeotools.lib import malib

import rasterio
from rasterio.plot import show

# -----------------------------------------------------------------------------
# class PlotLib
#
# This class provides plotting functions (e.g., scatter, histogram, maps).
# -----------------------------------------------------------------------------
class PlotLib(object):

    # -------------------------------------------------------------------------
    # __init__
    # -------------------------------------------------------------------------
    def __init__(self, debugLevel, histogramPlot, scatterPlot, fitPlot):
        self._debugLevel = debugLevel
        self._histogramPlot = histogramPlot
        self._scatterPlot = scatterPlot
        self._fitPlot = fitPlot
        return

    # -------------------------------------------------------------------------
    # trace()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def trace(self, value, override=False):
        if ((self._debugLevel >= 3) or override == True):
            print(value)

    # -------------------------------------------------------------------------
    # plot_maps()
    #
    # Generate and display image maps  for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_maps(self, masked_array_list, fn_list, figsize=(10, 5), title='Reflectance (%)',
                  cmap_list=['RdYlGn', 'RdYlGn'], override=False):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        :param cmap_list:
        :param override:
        """
        if (((self._debugLevel >= 2) and (self._histogramPlot == True)) or override == True):
            fig, axa = plt.subplots(nrows=1, ncols=len(fn_list), figsize=figsize, sharex=False, sharey=False)
            for i, ma in enumerate(masked_array_list):
                f_name = fn_list[i]
                divider = make_axes_locatable(axa[i])
                cax = divider.append_axes('right', size='2.5%', pad=0.05)
                im1 = axa[i].imshow(ma, cmap=cmap_list[i], clim=malib.calcperc(ma, perc=(1, 95)))
                cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend='max')
                axa[i].set_title(os.path.split(f_name)[1], fontsize=10)
                cb.set_label(title)

                plt.tight_layout()

    # -------------------------------------------------------------------------
    # plot_histograms()
    #
    # Generate and display histograms for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_histograms(self, masked_array_list, fn_list, figsize=(10, 3),
                        title="WARPED MASKED ARRAY", override=False):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        :param override:
        """
        if (((self._debugLevel >= 2) and (self._histogramPlot == True)) or override == True):
            fig, axa = plt.subplots(nrows=1, ncols=len(masked_array_list), figsize=figsize, sharex=True, sharey=True)

            for i, ma in enumerate(masked_array_list):
                f_name = os.path.split(fn_list[i])[1]
                self.trace(f" {ma.count()} valid pixels in {title} version of {f_name}")

                h = axa[i].hist(ma.compressed(), bins=512, alpha=0.75)
                axa[i].set_title(title + ' ' + f_name, fontsize=10)

            plt.tight_layout()

    # -------------------------------------------------------------------------
    # plot_scatter()
    #
    # Generate and display scatter plots for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_scatter(self, x_data, y_data, title="Raster Data Scatter Plot", null_value=-10,
                     override=False):
        """

        :param x_data:
        :param y_data:
        :param title:
        :param null_value:
        :param override:
        """
        if (((self._debugLevel >= 2) and (self._scatterPlot == True)) or override == True):
            plt.rcParams["font.family"] = "Times New Roman"
            # Declaring the figure, and hiding the ticks' labels
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            # Actually Plotting the data
            plt.scatter(x_data, y_data, s=0.1, c='black')
            # Making the graph pretty and informative!
            plt.title(title, fontsize=28)
            plt.xlabel("X-Axis Raster", fontsize=22)
            plt.ylabel("Y-Axis Raster", fontsize=22)
            plt.show()

    # -------------------------------------------------------------------------
    # plot_fit()
    #
    # Generate and display scatter and line fit for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_fit(self, x, y, slope, intercept, override=False):
        """

        :param x:
        :param y:
        :param slope:
        :param intercept:
        :param override:
        """
        if (((self._debugLevel >= 2) and (self._fitPlot == True)) or override == True):
            print(ggplot()  # What data to use
                  # + aes(x="date", y="pop")  # What variable to use
                  + aes(x=x, y=y)  # What variable to use
                  + geom_bin2d(binwidth=10)  # Geometric object to use for drawing
                  + geom_abline(slope=slope, intercept=intercept, size=2)
                  + geom_smooth(method='lm', color='red'))
            # + xlim(0,500) + ylim(0,500)

    # -------------------------------------------------------------------------
    # plot_combo()
    #
    # Generate and display histograms for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_combo(self, fname, figsize=(10, 3),
                        title="WARPED MASKED ARRAY", override=False):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        :param override:
        """
        from rasterio.plot import show_hist
        from matplotlib import pyplot
        if (((self._debugLevel >= 2) and (self._histogramPlot == True)) or override == True):
            imageSrc = rasterio.open(fname)
            fig, (axrgb, axhist) = pyplot.subplots(1, 2, figsize=figsize)
            show(imageSrc, ax=axrgb)
            show_hist(imageSrc, bins=50, histtype='stepfilled',lw=0.0, stacked=False, alpha=0.3, ax=axhist)
            pyplot.show()
