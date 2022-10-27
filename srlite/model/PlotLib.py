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
    def __init__(self, debug_level):
        self._debug_level = debug_level
        return

    # -------------------------------------------------------------------------
    # trace()
    #
    # Print trace debug (cus
    # -------------------------------------------------------------------------
    def trace(self, value):
        if (self._debug_level > 0):
            print(value)

    # -------------------------------------------------------------------------
    # plot_compare()
    #
    # Generate and display image maps for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_compare(self, evhr_pre_post_ma_list, compare_name_list):

        figsize = (5, 3)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize, sharex=True, sharey=True)
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        for i, ma in enumerate(evhr_pre_post_ma_list):
            f_name = compare_name_list[i]

            h = ax.hist(ma.compressed(), bins=256, alpha=0.5, label=f_name, color=colors[i])
            ax.legend()
            ax.set_xlim((0, max([ma.mean() + 3 * ma.std() for ma in evhr_pre_post_ma_list])))
            ax.set_xlabel('Reflectance (%)', fontsize=12)

        plt.tight_layout()

    # -------------------------------------------------------------------------
    # plot_maps()
    #
    # Generate and display image maps for 2-dimensional list of masked arrays
    # -------------------------------------------------------------------------
    def plot_maps(self, masked_array_list, fn_list, figsize=(10, 5), title='Reflectance (%)',
                  cmap_list=['RdYlGn', 'RdYlGn']):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        :param cmap_list:
        :param override:
        """
        if (self._debug_level >= 2):
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
                        title="WARPED MASKED ARRAY"):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        :param override:
        """
        if (self._debug_level >= 2):
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
    def plot_scatter(self, x_data, y_data, title="Raster Data Scatter Plot",
                     null_value=-10):
        """

        :param x_data:
        :param y_data:
        :param title:
        :param null_value:
        :param override:
        """
        if (self._debug_level >= 2):
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
    def plot_fit(self, x, y, slope, intercept):
        """

        :param x:
        :param y:
        :param slope:
        :param intercept:

        """
        if (self._debug_level >= 2):
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
                   title="WARPED MASKED ARRAY"):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        """
        from rasterio.plot import show_hist
        from matplotlib import pyplot
        imageSrc = rasterio.open(fname)
        if (self._debug_level >= 2):
            self.plot_combo_array(imageSrc, figsize, title)

    def plot_combo_array(self, imageSrc, figsize=(10, 3),
                         title="WARPED MASKED ARRAY"):
        """

        :param masked_array_list:
        :param fn_list:
        :param figsize:
        :param title:
        """
        from rasterio.plot import show_hist
        from matplotlib import pyplot
        if (self._debug_level >= 2):
            fig, (axrgb, axhist) = pyplot.subplots(1, 2, figsize=figsize)
            show(imageSrc, ax=axrgb)
            show_hist(imageSrc, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3, ax=axhist,
                      title=title)
            pyplot.show()

    def plot_maps2(self, masked_array_list, names_list, figsize=None, cmap_list=None, clim_list=None, title_text=""):
        if figsize is None:
            figsize = (len(names_list) * 7, 5)

        fig, axa = plt.subplots(nrows=1, ncols=len(masked_array_list), figsize=figsize, sharex=False, sharey=False)

        for i, ma in enumerate(masked_array_list):

            if cmap_list is None:
                cmap = 'RdYlGn'
            else:
                cmap = cmap_list[i]

            if clim_list is None:
                clim = malib.calcperc(ma, perc=(1, 95))
            else:
                clim = clim_list[i]

            f_name = names_list[i]

            divider = make_axes_locatable(axa[i])
            cax = divider.append_axes('right', size='2.5%', pad=0.05)
            im1 = axa[i].imshow(ma, cmap=cmap, clim=clim)
            cb = fig.colorbar(im1, cax=cax, orientation='vertical', extend='max')
            axa[i].set_title(title_text + os.path.split(f_name)[1], fontsize=10)
            cb.set_label('Reflectance (%)')

            plt.tight_layout()

    def plot_hists2(self, masked_array_list, names_list, figsize=None, title_text=""):
        if figsize is None:
            figsize = (len(names_list) * 7, 5)

        fig, axa = plt.subplots(nrows=1, ncols=len(masked_array_list), figsize=figsize, sharex=False, sharey=False)

        for i, ma in enumerate(masked_array_list):
            f_name = names_list[i]
            print(f" {ma.count()} valid pixels in INPUT MASKED ARRAY version of {f_name}")

            h = axa[i].hist(ma.compressed(), bins=256, alpha=0.75)
            axa[i].set_title(title_text + os.path.split(f_name)[1], fontsize=10)

        plt.tight_layout()
