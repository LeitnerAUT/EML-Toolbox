import eml_toolbox as eml;
#from scipy.ndimage.filters import uniform_filter1d;
import numpy as np;
import glob;
from matplotlib import pyplot as plt;
from IPython import get_ipython
_ipython = get_ipython();

#%% first close everything:
plt.close("all");

#%% define parameters:
    
# define search string to find images:
folder_search_str = "demo_images/*.tif*"
# define image no. in folder to load:
image_no = 2
# define the angle vector:
angle_vector_deg = np.arange(0, 360, step = 1);
angle_vector_rad = np.deg2rad(angle_vector_deg);

#%% get content of directory:
image_list = glob.glob(folder_search_str);
image_file = image_list[image_no-1];

# load image: 
image_data = eml.ed.load_image(image_file);

# crop image if needed:
#image_data = image_data[5:-100,150:-150]

# test whether surface-tension image or density image:
check_invert = eml.ed.check_image_invert(image_data)
#check_invert = True;
if(check_invert):
    image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!

#%% coarse edge detection:
debug_data = {}
resize_factor = 1
coarse_edge_call_method = lambda x: eml.ed.edge_coarse.process_image(image_data,
                                                                     angle_vector_rad,
                                                                     resize_factor = resize_factor,
                                                                     method = x,
                                                                     #threshold = 180,
                                                                     edge_filter_function_call = [eml.ed.filter_sobel,{}],
                                                                     edge_threshold = [200, np.inf],
                                                                     #edge_pixels_cut_off_limits = [0.90, 1.10],
                                                                     debug = True)
    
centroid_x, centroid_y, pixel_count, radius_approx_vector_spline, function_parameters, debug_data["spline"] = coarse_edge_call_method("spline");

#[coarse_edge_call_method("fit") for i in range(1000)]
#eml.various.stop();

image_data_edge = debug_data["spline"]["image_data_edge"];
edge_pixels_coordinates = debug_data["spline"]["edge_pixels_coordinates"];
edge_pixels_angles_sort = debug_data["spline"]["edge_pixels_angles_sort"];
edge_pixels_radii_sort = debug_data["spline"]["edge_pixels_radii_sort"];
edge_pixels_values_sort = debug_data["spline"]["edge_pixels_values_sort"];
threshold = debug_data["spline"]["threshold"]

_, _, _, radius_approx_vector_fit, _, debug_data["fit"] = coarse_edge_call_method("fit");        
_, _, _, radius_approx_vector_legendre, _, debug_data["legendre"] = coarse_edge_call_method("legendre");
_, _, _, radius_approx_vector_mvgAvg, _, debug_data["mvgAvg"] = coarse_edge_call_method("mvgAvg");
_, _, _, radius_approx_vector_mvgAvg_weighted, _, debug_data["mvgAvg_weighted"] = coarse_edge_call_method("mvgAvg_weighted");
_, _, _, radius_approx_vector_legendre_opt, _, debug_data["legendre_optimized"] = coarse_edge_call_method("legendre_optimized");

plt.figure();
plt.plot(np.rad2deg(edge_pixels_angles_sort), edge_pixels_radii_sort, '.');
plt.plot(angle_vector_deg, radius_approx_vector_mvgAvg, "-", color = "C1", label="mvAvg");
plt.plot(angle_vector_deg, radius_approx_vector_mvgAvg_weighted, "--", color = "C1", label="weighted mvAvg");
plt.plot(angle_vector_deg, radius_approx_vector_spline, "-", color = "C2", label="mvgAvg + spline");
plt.plot(angle_vector_deg, radius_approx_vector_legendre, "-", color = "C3", label="legendre");
plt.plot(angle_vector_deg, radius_approx_vector_legendre_opt, "--", color = "C3", label="legendre optimized")
plt.plot(angle_vector_deg, radius_approx_vector_fit, "-", color = "C4", label="Polyfit");
plt.legend();
plt.xlabel("Angle / degree");
plt.ylabel("Radius / px");

plt.figure(figsize=(12,10));
plt.subplot(211)
plt.imshow(image_data_edge, cmap="gray");
plt.plot(resize_factor*(centroid_x + edge_pixels_radii_sort*np.cos(edge_pixels_angles_sort)), 
         resize_factor*(centroid_y - edge_pixels_radii_sort*np.sin(edge_pixels_angles_sort)),'.',
         label="Edge pixels")
plt.legend();

for i in range(2):
    plt.subplot(2,2,i+3)
    if i == 0:
        plt.imshow(image_data_edge, cmap="gray");
    else:
        plt.imshow(image_data, cmap="gray");
    plt.plot(resize_factor*(centroid_x + radius_approx_vector_mvgAvg*np.cos(angle_vector_rad)), 
             resize_factor*(centroid_y - radius_approx_vector_mvgAvg*np.sin(angle_vector_rad)),
             '--',color = "C1", label="mvgAvg")
    plt.plot(resize_factor*centroid_x + resize_factor*radius_approx_vector_mvgAvg_weighted*np.cos(angle_vector_rad), 
             resize_factor*centroid_y - resize_factor*radius_approx_vector_mvgAvg_weighted*np.sin(angle_vector_rad),'.',
             color = "C1", label="weighted mvgAvg")
    plt.plot(resize_factor*centroid_x + resize_factor*radius_approx_vector_spline*np.cos(angle_vector_rad), 
             resize_factor*centroid_y - resize_factor*radius_approx_vector_spline*np.sin(angle_vector_rad),'--',
             color = "C2", label="mvgAvg + spline")
    plt.plot(resize_factor*centroid_x + resize_factor*radius_approx_vector_legendre*np.cos(angle_vector_rad), 
             resize_factor*centroid_y - resize_factor*radius_approx_vector_legendre*np.sin(angle_vector_rad),'--',
             color = "C3", label="legendre")
    plt.plot(resize_factor*centroid_x + resize_factor*radius_approx_vector_legendre_opt*np.cos(angle_vector_rad), 
             resize_factor*centroid_y - resize_factor*radius_approx_vector_legendre_opt*np.sin(angle_vector_rad),'.',
             color = "C3", label="legendre optimized")
    plt.plot(resize_factor*centroid_x + resize_factor*radius_approx_vector_fit*np.cos(angle_vector_rad), 
             resize_factor*centroid_y - resize_factor*radius_approx_vector_fit*np.sin(angle_vector_rad),'--',
             color = "C4", label="Polyfit")
    plt.legend();

#%% plot reference (cubic fit method):
_, _, _, radii_vector_fit_cubic, _, _ = \
    eml.ed.edge_fit_cubic.process_image(image_data,
                                        angle_vector_rad,
                                        #angle_filter_function = uniform_filter1d,
                                        #angle_filter_window_size_rad = np.deg2rad(10),
                                        #edge_filter_threshold = 0.2
                                        )
    
_, _, _, radii_vector_fit_gaussian, _, _ = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                            angle_vector_rad,
                                            )

_, _, _, radii_vector_fit_gaussian_new, _, _ = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                            angle_vector_rad,
                                            method = "coarse_edge_first",
                                            )
#%% compare:
plt.figure(figsize=(12,10));
plt.plot(angle_vector_deg, radii_vector_fit_cubic, label="reference (subpixel cubic fit)", color="C0");
plt.plot(angle_vector_deg, radius_approx_vector_mvgAvg, "-", color = "C1", label="mvgAvg");
plt.plot(angle_vector_deg, radius_approx_vector_mvgAvg_weighted, "--", color = "C1", label="weighted mvgAvg");
plt.plot(angle_vector_deg, radius_approx_vector_spline, "-", color = "C2", label="mvgAvg + spline");
plt.plot(angle_vector_deg, radius_approx_vector_legendre, "-", color = "C3",  label="legendre");
plt.plot(angle_vector_deg, radius_approx_vector_legendre_opt, "--", color = "C3",  label="legendre optimized")
plt.plot(angle_vector_deg, radius_approx_vector_fit, "-", color="C4", label="Polyfit");
plt.plot(angle_vector_deg, radii_vector_fit_gaussian, "-", color="C5", label="gaussian")
plt.plot(angle_vector_deg, radii_vector_fit_gaussian_new, "--", color="C5", label="gaussian new")
plt.xlabel("Angle / degree");
plt.ylabel("Radius / px");
plt.legend();

#%% show plots (if not run within ipython console)
plt.show()
