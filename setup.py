#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='srlite',
    version='1.1.0',
    description='Libraries and command-line utilities for surface reflectance',
    author=Glenn Tamkin',
    author_email='glenn.s.tamkin@nasa.gov',
    license='MIT',
    url='https://github.com/nasa-nccs-hpda/srlite',
    packages=['srlite'],
    long_description=open('README.md').read(),
    install_requires=['pygeotools','pandas','osgeo','numpy','scipy','matplotlib','pylr2','plotnine','rasterio','mpl_toolkits','sklearn'],
)