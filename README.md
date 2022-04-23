# Finlon_et_al_2021_DFR
Repository for the manuscript: Investigation of Microphysical Properties within Regions of Enhanced Dual-Frequency Ratio During the IMPACTS Field Campaign (Under Review as of January 2022)

This repository contains the core scripts to carry out analysis in the study, a few sample datasets, and a notebook tutorial working with the sample datasets. Below is a description of each file in the repository.

1. ```Tutorial.ipynb```: A tutorial notebook demonstrating a workflow using some sample datasets.  

2. ```p3.py```: Python module containing routines to combine 2D-S and HVPS datasets (processed by the UIOOPS package) and compute bulk microphysical properties.  

3. ```forward.py```: Python module that creates a class of forward-simulated Z at W-, Ka-, Ku-, and X-band for various degrees of riming.  

4. ```ess238-sup-0002-supinfo.tex```: Scattering tables from the Leinonen and Szyrmer (2015; https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2015EA000102) study.  

5. ```er2.py```: Python module contianing routines to read the ER-2 radar data (https://ghrc.nsstc.nasa.gov/uso/ds_details/collections/impactsC.html) and post-process the data.  

6. ```dfr_enhancement.py```: Python module containing a routine to determine regions of enhanced DFR based on the radar-matched DFR values.  

7. ```match.py```: Python module containing routines to match ER-2 radar data and outputs from the Chase et al. (2021; https://doi.org/10.1175/JAMC-D-20-0177.1) radar retrieval to the P-3 location.  

8. ```psd.2ds.20200207_example.nc```: Sample dataset containing binned and bulk microphysical properties from the 2D-S. (processed by UIOOPS; https://github.com/joefinlon/UIOPS)  

9. ```psd.hvps.20200207_example.nc```: Sample dataset containing binned and bulk microphysical properties from the HVPS.  

10. ```hiwrap.matched.20200207_example.nc```: Sample dataset containing HIWRAP Z (Ku and Ka band) matched to the P-3 aircraft.