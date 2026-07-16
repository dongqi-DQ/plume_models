[![DOI](https://zenodo.org/badge/1216454600.svg)](https://doi.org/10.5281/zenodo.21388029)


These Python scripts are based on the zero buoyancy plume model by Singh and O'Gorman (2013) and the spectral plume model by Zhou and Xie (2019).
## Matlab version:

- ZBP matlab code: `zero_buoyancy_plume.m` Singh and O'Gorman (2013)
- Spectral plume model (SPM) matlab code: `plume_model.m` Zhou and Xie (2019)

## Python version:

- ZBP: `zbp.py`; the script saves a figure and a CSV with vertical profiles of all outputs by default
- SPM: `spectral_plume_model.py`; can chooose `model_type` between `zero-buoyancy` and or `spectral`; the script saves a figure and a CSV with vertical profiles of all outputs by default.  
- Precipitation added: `precip_ext.py`; user can input LCL height (m) and precipitation to adjust the model; the script saves a figure and a CSV with vertical profiles of all outputs by default.

All Python scripts have their own jupyter notebook version (ipynb).

### Plotting
The plotting notebook (plotting.ipynb) create plots in the Palmer plane using the LCL and precipitation based entrainment rate. 

Plotting scripts from Marty Singh are `MSplot_Palmer_plane.ipynb` and `MSplot_profiles.ipynb`.


### Python environment and packages
The scipy function for plotting may require `scipy==1.16.2`. A yml file is added if one wants to create the entire environment.

I used the [cmc colormap pacakge](https://github.com/callumrollo/cmcrameri) for some figures.
