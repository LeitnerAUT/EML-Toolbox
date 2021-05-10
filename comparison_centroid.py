import eml_toolbox as eml;
from scipy.ndimage import center_of_mass;
import glob;
from matplotlib import pyplot as plt
from IPython import get_ipython
_ipython = get_ipython();

# only uncomment the line, if a compiler is installed to compile the cython 
# functions!
#import eml_toolbox._edge_detection._core._centroid_cython as eml_cython;

#%% set parameters:
# parameters for performance comparison:
nruns = 3; # number of runs
nloops = 50; # number of loops per run

# define search string to find images:
folder_search_str = "demo_images/*.tif*"
# define image no. in folder to load:
image_no = 6

#%% get content of directory:
image_list = glob.glob("demo_images/*.tif*");
image_file = image_list[image_no-1];

#%% Some first output:
print("Processing image: %s\n" % image_file)

#%% load image: 
image_data = eml.ed.load_image(image_file);

# test whether surface-tension image or density image:
check_invert = eml.ed.check_image_invert(image_data)
if(check_invert):
    image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!

#%% centroid part:
# automatically determine threshold and check consistency of all variants: 
auto_threshold_borders, _, _ = eml.ed.auto_threshold_borders(image_data, border_width = 30);
auto_threshold = eml.ed.auto_threshold(image_data);

# only execute if module eml_cython was loaded:
if "eml_cython" in dir():
    auto_threshold_cython = eml_cython.auto_threshold_cython(image_data);

# now determine centroid and check consistency of all variants (all use the 
# same threshold value): 
centroid_x, centroid_y, pixel_count_numpy = eml.ed.centroid(image_data, auto_threshold_borders);
# only execute if module eml_cython was loaded:
if "eml_cython" in dir():
    centroid_x_cython, centroid_y_cython, pixel_count_cython = eml_cython.centroid_cython(image_data, auto_threshold_borders);

# compare with scipy-version, returns center of mass in image coordinates / 
# array coordinates (1st index: row, 2nd index: column):
centroid_y_scipy, centroid_x_scipy = center_of_mass(image_data > auto_threshold_borders);

#%% show result:
plt.figure()
plt.imshow(image_data, cmap="gray")
plt.plot(centroid_x, centroid_y, marker="o", label="Centroid")
plt.legend()

#%% show plots (if not run within ipython console)
plt.show()

#%% compare performance of different centroid routines:
print("auto-threshold: ", end=""); _ipython.magic("timeit -n %i -r %i eml.ed.centroid_auto_threshold(image_data)" % (nloops, nruns));
print("auto-threshold borders: ", end=""); _ipython.magic("timeit -n %i -r %i eml.ed.centroid_auto_threshold_borders(image_data, border_width=30)" % (nloops, nruns));
# only execute if module eml_cython was loaded:
if "eml_cython" in dir():
    print("auto-threshold borders: ", end=""); _ipython.magic("timeit -n %i -r %i eml.ed.centroid_auto_threshold_borders(image_data, border_width=30, centroid_function=eml_cython.centroid_cython)" % (nloops, nruns));
    