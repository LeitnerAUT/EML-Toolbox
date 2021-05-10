# EML-Toolbox

This is the electromagnetic levition (EML) toolbox, work in progress!

This repository contains the actual EML-Toolbox for Python (subfolder [eml_toolbox](./eml_toolbox), 
hereinafter referred shortly as "toolbox") as well as some demo scripts and data 
on the root level of this repository that demonstrate the usage of the toolbox. 

## Requirements (see requirements.txt):
The toolbox requires mainly the following packages, which can be usually installed 
easily in any Python distribution (e.g. Anaconda). The packages marked as optional 
are not directly used by the toolbox but imported in some of the demo scripts where 
different methods are compared and results are presented:
* numpy
* opencv
* scipy
* matplotlib (optional)
* ipython (optional)
* joblib (optional)
* scikit-image (optional)
* imageio (optional)

## Table of contents:
Here is a small description of the files and folders in this repository:

| name | content |
| ----------- | ------- |
| Folders:    |         |
| [demo_data](./demo_data)   | folder containing demo data for the Legendre fitting procedure. The data in this folder was created by running the [demo_process_folder.py](./demo_process_folder.py) script on images from the [demo_images](./demo_images) folder |
| [demo_images](./demo_images) | folder containing demo image sequences for testing the edge detection programs. The names of the sub folders describe the origin (facility) of the demo images |
| [eml_toolbox](./eml_toolbox) | the actual EML-toolbox which is used in the demo programs and which can be imported in own programs to use the various routines |
| Files:     |         |
| [comparison_centroid.py](./comparison_centroid.py) | script that shows a comparison of the different (threshold-based) centroid routines. They all give identical results, but performance differs.|
| [comparison_edge_detection.py](./comparison_edge_detection.py) | runs the different edge detection methods on a single image and shows a performance comparison. |
| [comparison_filters.py](./comparison_filters.py) | script showing a comparison of different edge filters. |
| [comparison_imread.py](./comparison_imread.py) | script that runs a performance comparison of imread routines of different python toolboxes to load images. |
| [demo_edge_detection_coarse.py](./demo_edge_detection_coarse.py) | demo program applying the coarse edge detection procedure to a selected image from the [demo_images](./demo_images) folder. |
| [demo_edge_detection_fit_cubic.py](./demo_edge_detection_fit_cubic.py) | demo program applying the cubic edge fitting procedure to a selected image from the [demo_images](./demo_images) folder. |
| [demo_edge_detection_fit_gaussian.py](./demo_edge_detection_fit_gaussian.py) | demo program applying the gaussian edge fitting procedure to a selected image from the [demo_images](./demo_images) folder. |
| [demo_edge_fitting_legendre.py](./demo_edge_fitting_legendre.py) | demo program that reads the data obtained from the edge fitting program in the folder [demo_data](./demo_data) and performs an optimized legendre fit to the radii data and from that, calculates the volume for each frame (line in the radii file). |
| [demo_process_folder.py](./demo_process_folder.py) | demo program that applies a selected edge-detection routine on a complete image sequences and writes the resulting information to a file in the folder [demo_data](./demo_data). |

## Exemplary parameters for different facilities:
Since the imaging system and thus the image quality varies for different levitation
facilities, some of the parameters of the various edge-detection routines need 
to be adapted specifically for each facility to optimize the result.

This section provides a list of exemplary parameters as a starting point for the 
various routines. Depending on the actual video and particular features of a video
(e.g. reflections on the sample, etc.), those parameters need further adaption. 

### JAXA:
The recommended edge-detection method is the edge_coarse method (due to its robustness)
* images need to be inverted manually, e.g. by setting the "check_invert" variable 
to true (automatic checking fails due to the shadowing at the image borders)
* manually set a crop mask for all images to crop the shadowed image border region, 
e.g. image_data = image_data[5:-100,150:-150]
* | edge dection method | parameters |
  | ------------------- | ---------- |    
  | edge_fit_cubic    | threshold = 180<br>edge_filter_add_tuple = (-3, +3) |
  | edge_fit_gaussian | to be completed |
  | edge_coarse       | threshold = 180<br>edge_threshold = 300<br>edge_pixels_cut_off_limits = [0.90, 1.10] |

    
### MSFC:
* to be completed:

### TU Graz side view:
* to be completed

### TU Graz top view:
* to be completed
    
### General hints:
* depending on the filename pattern of the images, the images may be sorted not 
in the correct order when the list of file names is generated via the glob.glob()
function in the demo files. 
To avoid that, a file name pattern with fixed length of the file number, e.g. image0001, 
image0002, image0003, etc. instead of image1, image2, image3 should be used.

## Usage:
Just import eml_toolbox in your script, e.g. import eml_toolbox as eml to have
access to the functions of the toolbox. 

## ToDo
- [ ] complete/check module- and function docstrings
  - [ ] especially for the [edge_fitting.py](./eml_toolbox/_edge_fitting.py) file
- [ ] complete exemplary parameter list
- [ ] create documentation (e.g. from docstrings via sphinx)
- [ ] expand [README.md](./README.md) file!
- [ ] check consistency of variable names
- [ ] check consistent use of color codings in plots and figures


## Brief documentation of the edge detection methods:

### edge_fit_cubic method:
The toolbox was started with this version. It is mainly based on the work by Bradshaw 
et al. [[1](#References)]. The basic concept is to fit the radial intensity profile with a cubic 
polynomial where the approximate edge is. The point of inflection of the fitted 
polynomial is then defined as the edge. 

This method gives by design sub-pixel precision for the determined edge. Compared
to threshold-based methods, it should be insensitive to intensity variations or 
non-uniform illuminations respectively. However, depending on the implementation, 
it can be sensitve to image errors or special features that can occur in ESL images, 
e.g. small reflections. 

The main drawback of this method turned out to be the poor performance (at least 
in Python), since for each radial search vector, the cubic polynomial fitting procedure 
has to be performed individually; a process that can not be vectorized due to the 
non-uniform length of the vectors of data points to fit. This was the motivation
to find other sub-pixel edge detection strategies that allow avoiding loops in 
python. 

### edge_fit_gaussian method:
Instead of fitting a cubic polynomial to the radial intensity profile at the approximate 
edge, the first derivative of the intensity profile is calculated and the edge is 
interpreted as a gaussian function with the peak at the sub-pixel edge detection.

The trick is not to fit a gaussian function to the first derivative of the intensity 
profile but just calulating the moments of the distribution, which is way faster 
since it is not an optimization problem and can be vectorized (inspired by [[2](#References)]).

### edge_coarse method:
This method was originally inspired by the idea to get a fast and reliable routine 
to determine the approximate edge by applying a derivative based edge detection 
filter and further analyze the obtained edge image. 

By applying such a derivatie based edge filter (e.g. sobel, kirsch), an edge image 
is obtained that contains a band of edge pixels where the intensity changes from 
bright to dark (or vice versa). For each edge pixel, the radius and the angle with 
respect to the centroid of the sample on the image are calculated. This data is 
then further filtered by using a moving average filter, a spline, high-order polynomial 
or linear combination of legendre polynomials. To speed up this method (since it 
was originally designed to get the approximate edge before continuing with a sub-pixel 
edge detection method), the image can be resized by specifying a resize factor.

But it turned out that this method is suitable to be used on its own (without resizing 
the image) and offers sub-pixel precision by design. Moreover, it is way less sensitive 
to image errors than other methods, since the number of edge pixels resulting from 
image errors is by orders of magnitude smaller than the number of "true" edge pixels
and thus will be "flattened" away by the applied filter (moving average, spline
polynomial, linear combination of legendre polynomials)  

Since this method relies on the result from applying the edge filter on the whole 
image and this step takes most of the computation time, the performance is only 
weakly dependent on the angle resolution for the radii detection. 

## References:
1. Bradshaw, R. C., Schmidt, D. P., Rogers, J. R., Kelton, K. F., & Hyers, R. W. (2005). Machine vision for high-precision volume measurement applied to levitated containerless material processing. Review of scientific instruments, 76(12), 125108. https://doi.org/10.1063/1.2140490
2. https://scipy-cookbook.readthedocs.io/items/FittingData.html#Fitting-gaussian-shaped-data