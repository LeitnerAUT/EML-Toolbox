import numpy as np;
import eml_toolbox as eml;
#import eml_toolbox._edge_detection._core._centroid_cython as eml_cython
import glob;
from IPython import get_ipython
_ipython = get_ipython();

#%% define parameters:
 
# parameters for performance comparison:
nruns = 7; # number of runs
nloops = 10; # number of loops per run    
 
# define search string to find images:
folder_search_str = "demo_images/*.tif*"
# define image no. in folder to load:
image_no = 8
# define the angle vector:
angle_vector_deg = np.arange(0, 360, step = 1);
angle_vector_rad = np.deg2rad(angle_vector_deg);

#%% get content of directory:
image_list = glob.glob(folder_search_str);
image_file = image_list[image_no-1];

#%% load image: 
image_data = eml.ed.load_image(image_file);

# test whether surface-tension image or density image:
check_invert = eml.ed.check_image_invert(image_data)
if(check_invert):
    image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!

#%% coarse edge detection:
call_edge_coarse = lambda x: eml.ed.edge_coarse.process_image(image_data,
                                                              angle_vector_rad,
                                                              method = x,
                                                              edge_filter_function_call = [eml.ed.filter_sobel,{}],
                                                              edge_threshold = 200)

#%% reference (subpixel cubic fit)
call_edge_cubic = lambda : eml.ed.edge_fit_cubic.process_image(image_data,
                                        angle_vector_rad,
                                        angle_filter_window_size_rad = np.deg2rad(10),
                                        edge_filter_threshold = 0.10
                                        )
#%% 
call_edge_gaussian_standard = lambda: eml.ed.edge_fit_gaussian.process_image(image_data, 
                                                                        angle_vector_rad);

call_edge_gaussian_coarse_first = lambda: eml.ed.edge_fit_gaussian.process_image(image_data, 
                                                                                 angle_vector_rad,
                                                                                 method = "coarse_edge_first",
                                                                                 step_division = 5,
                                                                                 angle_filter_window_size_rad = np.deg2rad(10),
                                                                                 threshold_grad = 0.1);

call_edge_gaussian_edge_image = lambda: eml.ed.edge_fit_gaussian.process_image(image_data, 
                                                                               angle_vector_rad,
                                                                               method = "edge_image",
                                                                               step_division = 5,
                                                                               angle_filter_window_size_rad = np.deg2rad(10));

#%% compare performance of different centroid routines:
print("reference (cubic fit): ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_cubic()'  % (nloops, nruns));
print("")
#%%
print("mvgAvg: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_coarse("mvgAvg")' % (nloops, nruns));
#%%
print("mvgAvg + Spline: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_coarse("spline")'  % (nloops, nruns));
print("")
#%%
print("legendre: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_coarse("legendre")'  % (nloops, nruns));
#%%
print("legendre_optimized: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_coarse("legendre_optimized")'  % (nloops, nruns));
print("")
#%%
print("fit: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_coarse("fit")'  % (nloops, nruns));
print("")
#%%
#print("weighted mvgAvg: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_coarse("mvgAvg_weighted")'  % (nloops, nruns));
#%%
print("gaussian fit: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_gaussian_standard()'  % (nloops, nruns));
#%%
print("gaussian fit new: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_gaussian_coarse_first()'  % (nloops, nruns));
#%%
print("gaussian edge image: ", end=""); _ipython.magic('timeit -n %i -r %i call_edge_gaussian_edge_image()'  % (nloops, nruns));

