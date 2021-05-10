import eml_toolbox as eml;
import numpy as np
from skimage import filters
import cv2;
import glob;
from matplotlib import pyplot as plt;
from IPython import get_ipython
_ipython = get_ipython();

#%% first close everything:
plt.close("all");

#%% set parameters:
# parameters for performance comparison:
nruns = 3; # number of runs
nloops = 25; # number of loops per run

# define search string to find images:
folder_search_str = "demo_images/*.tif*"
# define image no. in folder to load:
image_no = 6
# define kernel size for filters:
kernel_size = 3;

#%% get content of directory:
image_list = glob.glob(folder_search_str);
image_file = image_list[image_no-1];

#%% Some first output:
print("Processing image: %s\n" % image_file)

#%% load image:
image_data = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE);

# create int16 and float arrays to test impact on filter performance:
image_data_int16 = image_data.astype(np.int16);
image_data_float32 = image_data.astype(np.float32);

# show how much time is consumed by converting array to int16 or float32. 
# NOTE: float16 is not supported by cv2!
print("astype(int16): ", end=""); _ipython.magic("timeit -n %i -r %i image_data.astype(np.int16)" % (nloops, nruns));
print("astype(float32): ", end=""); _ipython.magic("timeit -n %i -r %i image_data.astype(np.float32)" % (nloops, nruns));
print("")

# test whether surface-tension image or density image:
check_invert = eml.ed.check_image_invert(image_data)
if(check_invert):
    image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!

#%% create short lambda functions:
edge_sobel = lambda image_data, kernel_size=-1: eml.ed.filter_sobel(image_data, kernel_size=-1)
edge_laplacian = lambda image_data: eml.ed.filter_laplacian(image_data, kernel_size)
edge_kirsch = lambda image_data: eml.ed.filter_kirsch(image_data)
edge_canny = lambda image_data: cv2.Canny(image_data, 0, 40, L2gradient=True)
edge_sobel_sk = lambda image_data: filters.sobel(image_data);
edge_scharr_sk = lambda image_data: filters.scharr(image_data);
edge_farid_sk = lambda image_data: filters.farid(image_data);

#%% compare performance of different filter routines:
print("sobel (uint8): ", end=""); _ipython.magic("timeit -n %i -r %i edge_sobel(image_data)" % (nloops, nruns));
print("sobel (int16): ", end=""); _ipython.magic("timeit -n %i -r %i edge_sobel(image_data_int16)" % (nloops, nruns));
print("sobel (float32): ", end=""); _ipython.magic("timeit -n %i -r %i edge_sobel(image_data_float32)" % (nloops, nruns));
print("")
print("laplacian (uint8): ", end=""); _ipython.magic("timeit -n %i -r %i edge_laplacian(image_data)" % (nloops, nruns));
print("laplacian (float32): ", end=""); _ipython.magic("timeit -n %i -r %i edge_laplacian(image_data_float32)" % (nloops, nruns));
print("")
print("kirsch (uint8): ", end=""); _ipython.magic("timeit -n %i -r %i edge_kirsch(image_data)" % (nloops, nruns));
print("kirsch (int16): ", end=""); _ipython.magic("timeit -n %i -r %i edge_kirsch(image_data_int16)" % (nloops, nruns));
print("kirsch (float32): ", end=""); _ipython.magic("timeit -n %i -r %i edge_kirsch(image_data_float32)" % (nloops, nruns));
print("")
print("canny: ", end=""); _ipython.magic("timeit -n %i -r %i edge_canny(image_data)" % (nloops, nruns));
print("")
print("skimage sobel (int16) ", end=""); _ipython.magic("timeit -n %i -r %i edge_sobel_sk(image_data_int16)" % (nloops, nruns));
print("skimage sobel (float32): ", end=""); _ipython.magic("timeit -n %i -r %i edge_sobel_sk(image_data_float32)" % (nloops, nruns));
print("skimage sobel: ", end=""); _ipython.magic("timeit -n %i -r %i edge_sobel_sk(image_data)" % (nloops, nruns));
print("")
print("skimage scharr (float32): ", end=""); _ipython.magic("timeit -n %i -r %i edge_scharr_sk(image_data_float32)" % (nloops, nruns));
print("skimage scharr: ", end=""); _ipython.magic("timeit -n %i -r %i edge_scharr_sk(image_data)" % (nloops, nruns));
print("")
print("skimage farid (float32): ", end=""); _ipython.magic("timeit -n %i -r %i edge_farid_sk(image_data_float32)" % (nloops, nruns));
print("skimage farid: ", end=""); _ipython.magic("timeit -n %i -r %i edge_farid_sk(image_data)" % (nloops, nruns));


#%% compare visually:   
plt.figure();
plt.subplot(2,1,1);
plt.imshow(image_data, cmap="gray");

# Sobel:
plt.subplot(2,7,8);
plt.imshow(edge_sobel((image_data_int16)), cmap="gray");
plt.title("Sobel");

# Laplacian:
plt.subplot(2,7,9);
plt.imshow(np.abs(edge_laplacian(image_data_int16)), cmap="gray");
plt.title("Laplacian");

# Kirsch:
plt.subplot(2,7,10);
plt.imshow(edge_kirsch(image_data_int16), cmap="gray");
plt.title("Kirsch");

# Canny:
plt.subplot(2,7,11);
plt.imshow(edge_canny(image_data), cmap="gray");
plt.title("Canny");

# Sobel skimage:
plt.subplot(2,7,12);
plt.imshow(edge_sobel_sk(image_data), cmap="gray");
plt.title("Sobel (Skimage)");

# Scharr skimage:
plt.subplot(2,7,13);
plt.imshow(edge_scharr_sk(image_data), cmap="gray");
plt.title("Scharr (Skimage)");

# Farid skimage:
plt.subplot(2,7,14);
plt.imshow(edge_farid_sk(image_data), cmap="gray");
plt.title("Farid (Skimage)");

#%% show plots (if not run within ipython console)
plt.show()