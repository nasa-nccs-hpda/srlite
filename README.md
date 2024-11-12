# Surface reflectance from commercial very high resolution multispectral imagery estimated empirically with synthetic Landsat 
Scientific analysis of changes of the Earth's land surface benefit from well-characterized, science quality remotely sensed data. This data quality is the result of models that estimate and remove atmospheric constituents and account for sun-sensor geometry.  Top-of-atmosphere (TOA) reflectance in commercial very high resolution (< 5 m; VHR) spaceborne imagery routinely varies for unchanged surface features because of signal variation from the combined effects of atmospheric haze and a range of sun-sensor geometric scenarios of acquisitions. Consistency from surface reflectance (SR) versions of this imagery must be sufficient to identify and track the change or stability of fine-scale features that, though small, may be widely distributed across remote domains, and serve as key indicators of critical broad-scale environmental change. Currently commercial SR products are available, but typically the model employed is proprietary and the costs for using these products over a large domain can be significant. We presented an open-source workflow for the scientific community for fine-scaled empirical estimation of surface reflectance from multispectral VHR imagery using reference from synthetically-derived coincident Landsat-based surface reflectance in Montesano et al. (2024) [1].  The $SR_{VHR}$ tool that sits at the end of this workflow [2], as well as the tools that precede it in this workflow, continue to evolve. 

#### Note: The most recent version of the $SR_{VHR}$ tool can be found at this repository, but development is underway on a software package that combines and presents these tools together as a toolkit.

## SR<sub>VHR</sub>: empirical estimation of VHR surface reflectance
The workflow for estimating surface reflectance for commercial VHR multispectral imagery (SR<sub>VHR</sub>).

![fig1_v3 (1)](https://github.com/user-attachments/assets/f3a6f82c-56bd-4b14-b3d2-74f55be47514)

References: 
1. **Surface Reflectance From Commercial Very High Resolution Multispectral Imagery Estimated Empirically With Synthetic Landsat (2023)**:  <em>Montesano et al. 2024 https://ieeexplore.ieee.org/document/10670299</em>
2. Preliminary [User Guide (October 2024)](https://github.com/nasa-nccs-hpda/srlite/blob/main/SRVHR%20-%20User's%20Guide%20and%20Workflow%20Documentation.md)

 Workflow Contributors | Role | Affiliation | 
| ---------------- | ---------------- | ---------------- |
| Paul M. Montesano |  Author ; Evaluator | NASA Goddard Space Flight Center Data Science Group |
| Matthew J. Macander |   Author ; Evaluator | Alaska Biological Research, Inc. |
| Jordan A. Caraballo-Vega  |  Developer | NASA Goddard Space Flight Center Data Science Group |
| Melanie J. Frost |  Author ; Evaluator | NASA Goddard Space Flight Center Data Science Group |
| Jian Li |  Developer | NASA Goddard Space Flight Center Data Science Group |
| Glenn S. Tamkin  |  Developer | NASA Goddard Space Flight Center Data Science Group |
| Mark L. Carroll |  PI | NASA Goddard Space Flight Center Data Science Group (Lead)|

