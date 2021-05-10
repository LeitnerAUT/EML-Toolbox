import eml_toolbox as eml;
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
angle_vector_deg = np.arange(0, 360, step = 5);
angle_vector_rad = np.deg2rad(angle_vector_deg);

threshold_grad = 0.1;

#%% get content of directory:
image_list = glob.glob(folder_search_str);
image_file = image_list[image_no-1];

# plot options:
# method to plot:
method = "standard";

# angles to plot:
step_size = int(30/(angle_vector_deg[1]-angle_vector_deg[0]));
angle_vector_plot = angle_vector_deg[::step_size];

# add custom angles if needed:
#angle_vector_plot = np.hstack((angle_vector_plot,158,246))

#%% load image: 
image_data = eml.ed.load_image(image_file);

# crop if necessary:
#image_data = image_data[5:-100,150:-150]

# test whether surface-tension image or density image:
check_invert = eml.ed.check_image_invert(image_data)
#check_invert = True
if(check_invert):
    image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!

# coarse edge for comparison:
centroid_x, centroid_y, pixel_count, radius_approx_vector, _, debug_data = \
    eml.ed.edge_coarse.process_image(image_data,
                                     angle_vector_rad,
                                     resize_factor = 1,
                                     method = "fit",
                                     edge_threshold = [200, np.inf]);

#%% fine edge detection:
radii_vector = {}
function_parameters = {}
_, _, _, radii_vector["standard"], function_parameters["standard"], debug_data["standard"] = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                           angle_vector_rad,
                                           threshold_grad = threshold_grad,
                                           debug=True);

_, _, _, radii_vector["standard_angleAvg"], function_parameters["standard_angleAvg"], debug_data["standard_angleAvg"] = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                           angle_vector_rad,
                                           angle_filter_window_size_rad = np.deg2rad(10),
                                           threshold_grad = threshold_grad,
                                           debug = True);

_, _, _, radii_vector["coarse_first"], function_parameters["coarse_first"], debug_data["coarse_first"] = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                           angle_vector_rad,
                                           method = "coarse_edge_first",
                                           step_division = 5,
                                           threshold_grad=threshold_grad,
                                           debug=True);

_, _, _, radii_vector["coarse_first_angleAvg"], function_parameters["coarse_first_angleAvg"], debug_data["coarse_first_angleAvg"] = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                           angle_vector_rad,
                                           method = "coarse_edge_first",
                                           step_division = 5,
                                           angle_filter_window_size_rad = np.deg2rad(10),
                                           threshold_grad = threshold_grad,
                                           debug = True);

_, _, _, radii_vector["edge_image"], function_parameters["edge_image"], debug_data["edge_image"] = \
    eml.ed.edge_fit_gaussian.process_image(image_data,
                                           angle_vector_rad,
                                           method = "edge_image",
                                           step_division = 5,
                                           angle_filter_window_size_rad = np.deg2rad(10),
                                           edge_threshold_search_limit = [0.96,1.02],
                                           threshold_grad = 0.25,
                                           debug = True);

#%% reference (subpixel cubic fit)
_, _, _, radii_vector_fit_cubic, _, _ = \
    eml.ed.edge_fit_cubic.process_image(image_data,
                                        angle_vector_rad,
                                        angle_filter_window_size_rad = np.deg2rad(10),
                                        edge_filter_threshold = 0.10
                                        )
#%% plot:
plt.figure(figsize=(12,10));
plt.imshow(image_data, cmap="gray");
plt.plot((centroid_x + radii_vector["standard"]*np.cos(angle_vector_rad)),
         (centroid_y - radii_vector["standard"]*np.sin(angle_vector_rad)),
         linewidth = 1,
         label="Gaussian fit (fast)");

plt.plot((centroid_x + radii_vector["standard_angleAvg"]*np.cos(angle_vector_rad)),
        (centroid_y - radii_vector["standard_angleAvg"]*np.sin(angle_vector_rad)),
        linewidth = 1,
        label="Gaussian fit (fast, angleAvg)");

plt.plot((centroid_x + radii_vector["coarse_first"]*np.cos(angle_vector_rad)),
        (centroid_y - radii_vector["coarse_first"]*np.sin(angle_vector_rad)),
        linewidth = 1,
        label="Gaussian fit (new)");

plt.plot((centroid_x + radii_vector["coarse_first_angleAvg"]*np.cos(angle_vector_rad)),
        (centroid_y - radii_vector["coarse_first_angleAvg"]*np.sin(angle_vector_rad)),
        linewidth = 1,
        label="Gaussian fit (new, angleAvg)");
plt.legend();

# plt.figure();
# plt.plot(angle_vector_deg, radii_vector["standard"], label="Gaussian fit (fast)", color="C0");
# plt.plot(angle_vector_deg, radii_vector["standard_angleAvg"], label="Gaussian fit (fast, angleAvg)", 
#          color="C1");
# plt.plot(angle_vector_deg, radii_vector["coarse_first"], label="Gaussian fit (new)", color="C2");
# plt.xlabel("Angle / deg");
# plt.ylabel("Radiux / px");
# plt.legend();

plt.figure(figsize=(12,10));
plt.plot(angle_vector_deg, radii_vector_fit_cubic, "--", color="C0", label = "reference");
plt.plot(angle_vector_deg, radii_vector["standard"], '-', color="C1", label="Standard");
plt.plot(angle_vector_deg, radii_vector["standard_angleAvg"], '--', color="C1", label = "Standard + angleAvg");
plt.plot(angle_vector_deg, radii_vector["coarse_first"], '-', color="C2", label = "Coarse edge first");
plt.plot(angle_vector_deg, radii_vector["coarse_first_angleAvg"], '--', color="C2", label = "Coarse edge first + angleAvg");
plt.plot(angle_vector_deg, radii_vector["edge_image"], '-', color="C3", label = "Use edge image");
plt.plot(angle_vector_deg, radius_approx_vector, '-', color="C4", label = "coarse mvgAvg+spline");

plt.xlabel("Angle / degree");
plt.ylabel("Radius / px");
plt.legend();



#%% plot for showing intensity profiles:
fit = lambda x, y0, mu, sigma : y0*np.exp(-(x-mu)**2/(2*sigma**2))    
    
mu_vector = debug_data[method]["mu_vector"]
sigma_vector = debug_data[method]["sigma_vector"]

r_vector = debug_data[method]["calc_edge"]["r_vector"]
if r_vector.ndim < 2:
    r_vector = np.tile(np.atleast_2d(r_vector).T,(1,mu_vector.size))
x_vector = debug_data[method]["calc_edge"]["x_vector"]
intensity_profiles_part = debug_data[method]["calc_edge"]["data"]
intensity_profiles_part_smooth = debug_data[method]["calc_edge"]["data_smooth"]
data_smooth_grad = debug_data[method]["calc_edge"]["data_smooth_grad"]
y0 = np.max(data_smooth_grad, axis=0);

# calculate number of subplot rows and columns to plot all angles. -1 for the 
# rows to get a wide-angle figure:
plot_rows = int(np.floor(np.sqrt(angle_vector_plot.size))) - 1;
plot_columns = int(np.ceil(angle_vector_plot.size/plot_rows));

# create the figure and subplots. The number of rows is multiplied by two, since
# each subplot should contain actually two plots about each other:
fig, ax = plt.subplots(2*plot_rows, plot_columns, figsize=(12,10))

# transpose and flatten the array for later access in the loop:
ax = (ax.T).ravel();

# loop over angles specified by angle_vector_plot:
for i, angle in enumerate(angle_vector_plot):
    # get index of current angle in angle_vector:
    angle_index = np.where(angle_vector_deg == angle)[0][0];
    
    # top subplots are every second element starting from the 1st element in 
    # the axis vector, there we plot the intensity profile related stuff:
    ax[0::2][i].plot(r_vector[:,angle_index],
                     intensity_profiles_part[:,angle_index],
                     label="raw data");
    ax[0::2][i].plot(r_vector[:,angle_index],
                     intensity_profiles_part_smooth[:,angle_index],
                     label='smoothed data');
    ax[0::2][i].plot((radii_vector[method][angle_index],radii_vector[method][angle_index]),
                     (0, 255),
                     label='edge fast');
    
    ax[1::2][i].plot(r_vector[:,angle_index],
                     data_smooth_grad[:,angle_index],
                     '.--',
                     label='smoothed data 1st derivative')
    ax[1::2][i].plot(r_vector[:,angle_index],
                     fit(x_vector, y0[angle_index], mu_vector[angle_index], sigma_vector[angle_index]), 
                     '--',
                     label='gauss')
    
# collect handles and labels for figure legend:
subplot1_legend_labels = ax[0::2][-1].get_legend_handles_labels();
subplot2_legend_labels = ax[1::2][-1].get_legend_handles_labels();

handles = np.append(subplot1_legend_labels[0], subplot2_legend_labels[0]);
labels = np.append(subplot1_legend_labels[1], subplot2_legend_labels[1]);

fig.legend(handles, labels, ncol = labels.size, loc='lower center');
fig.suptitle("method: %s" % method)

# tight layout takes care of avoiding overlapping of axis labels, etc.
# height is reduced so that there is a free space at the top to contain the 
# figure legend:
fig.tight_layout(rect=(0,0.05,1,0.95)); 

#%% show plots (if not run within ipython console)
plt.show()