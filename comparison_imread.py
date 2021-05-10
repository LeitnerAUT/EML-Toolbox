import cv2;
import glob;
import imageio;
from skimage.io import imread as sk_imread;
from matplotlib import pyplot as plt
from IPython import get_ipython
_ipython = get_ipython();

#%% set parameters:
# parameters for performance comparison:
nruns = 3; # number of runs
nloops = 10; # number of loops per run

# define search string to find images:
folder_search_str = "demo_images/*.tif*"
# define image no. in folder to load:
image_no = 6;

#%% get content of directory:
image_list = glob.glob(folder_search_str);
image_file = image_list[image_no-1];

# define short lambda functions for the different imread routines: 
imread_sk = lambda: sk_imread(image_file, as_gray=True);
imread_imageio = lambda: imageio.imread(image_file, format='tiff');
imread_cv2 = lambda: cv2.imread(image_file,cv2.IMREAD_GRAYSCALE);

# load image once to get rid of any disk delays: 
imread_sk()
imread_imageio()
imread_cv2()
    
#%% Output:
print("Processing image: %s\n" % image_file)

plt.figure()
plt.imshow(imread_cv2(), cmap="gray")

#%% show plots (if not run within ipython console)
plt.show()

#%% compare performance of different imread routines:
# obviously, imageio.imread gets very slow if a compressed tiff image is loaded:
# imageio is only on image_no 1 fastest, otherwise way slower (orders of magnitude!!!) 
print("skimage.io.imread: ", end=""); _ipython.magic("timeit -n %i -r %i imread_sk()" % (nloops, nruns));
print("imageio.imread: ", end=""); _ipython.magic("timeit -n %i -r %i imread_imageio()" % (nloops, nruns));
print("cv2.imread: ", end=""); _ipython.magic("timeit -n %i -r %i imread_cv2()" % (nloops, nruns));