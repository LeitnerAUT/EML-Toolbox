import numpy as _np;

#%% import cython functions:
# the following two lines are needed to include the numpy headers when the
# pyx file is compiled. Please see https://gist.github.com/hannes-brt/757191
import pyximport as _pyximport; 
_pyximport.install(setup_args={'include_dirs': _np.get_include()})

# import the functions available as cython version:
from ._cython_functions import auto_threshold_cython, centroid_cython