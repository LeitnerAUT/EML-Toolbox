import eml_toolbox as eml;
import numpy as np;
import glob;
from scipy.interpolate import UnivariateSpline;
from matplotlib import pyplot as plt;

#%% first close everything:
plt.close("all");

#%% define parameters:
    
# define search string to find images:
folder_search_str = "demo_images/*.tif*"
# define image no. in folder to load:
image_no = 2
# define the step size for the angle vector:
angle_vector_deg_step_size = 1;

angle_vector_deg = np.arange(0, 360, step = 1);
angle_vector_rad = np.deg2rad(angle_vector_deg);

#%% get content of directory:
image_list = glob.glob(folder_search_str);
image_file = image_list[image_no-1];

#%% load image: 
image_data = eml.ed.load_image(image_file);

# test whether surface-tension image or density image:
check_invert = eml.ed.check_image_invert(image_data)

# manually invert images if automatic check fails:
# check_invert = True
if(check_invert):
    image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!

# crop image if needed:
# image_data = image_data[5:-100,150:-150]

# get edges:
centroid_x, centroid_y, pixel_count, radii, function_parameters, debug_data = \
    eml.ed.edge_fit_cubic.process_image(image_data,
                                        angle_vector_rad,
                                        #threshold = 180,
                                        debug = True,
                                        edge_filter_add_tuple = (-2, +2)
                                        #angle_filter_window_size_rad = np.deg2rad(5),
                                        #edge_filter_threshold = 0.10
                                        )

result_threshold = debug_data["radii_threshold_vector"];

#%% show some plots for different angles:
# extract from debug data:
radius_vector = debug_data["radius_vector"];
    
angle_vector_plot = angle_vector_deg[::30];

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
    
    # get radius for current angle
    radius = radii[angle_index];
    
    # extract data from debug variable for current angle:
    intensity_profile = debug_data["intensity_profiles_array"][:,angle_index];
    edge_profile = debug_data["edge_profiles_array"][:,angle_index];
    approx_edge_threshold = debug_data["radii_threshold_vector"][angle_index];
    radius_vector_fit = debug_data["radii_vector_fit_all_list"][angle_index];
    intensity_vector_fit = debug_data["intensity_vector_fit_all_list"][angle_index];
    fit_coeff = debug_data["fit_coeff_all_list"][angle_index];
    
    # top subplots are every second element starting from the 1st element in 
    # the axis vector, there we plot the intensity profile related stuff:
    ax[0::2][i].plot(radius_vector, intensity_profile);
    ax[0::2][i].set_title("Angle: %.0i degree" % angle_vector_deg[angle_index]);
    ax[0::2][i].set_xlim((approx_edge_threshold - 20, approx_edge_threshold + 20));
    ax[0::2][i].set_ylim((-10,260));
    ax[0::2][i].set_xlabel("Radius / px");
    ax[0::2][i].set_ylabel("Pixel intensity / -");
    
    if(radii[i] != np.NaN):
        ax[0::2][i].plot(radius_vector_fit, intensity_vector_fit, label="data to fit")
        ax[0::2][i].plot(radius_vector_fit, 
                  np.polyval(fit_coeff,radius_vector_fit-np.mean(radius_vector_fit)) + np.mean(intensity_vector_fit),
                  '-r', label="fitted polynomial");
        ax[0::2][i].plot((radius, radius),(np.min(intensity_vector_fit), np.max(intensity_vector_fit)),'--r', label="Edge (Polyfit)")
        ax[0::2][i].plot((approx_edge_threshold,approx_edge_threshold), (0,255), label = "Edge threshold");
    
    # bottom subplots are every second element starting from the 2nd element in 
    # the axis vector, there we plot the edge profile related stuff:
    ax[1::2][i].plot(radius_vector, edge_profile,'-b', label="edge_filter pixel intensity");
    ax[1::2][i].plot(radius_vector_fit, 
              edge_profile[np.in1d(radius_vector, radius_vector_fit)],
              '.r',label="Fit area");
    ax[1::2][i].set_xlim(approx_edge_threshold - 20, approx_edge_threshold + 20);
    ax[1::2][i].set_xlabel("Radius / px");
    ax[1::2][i].set_ylabel("Edge filter pixel intensity / -");

# collect handles and labels for figure legend:
subplot1_legend_labels = ax[0::2][-1].get_legend_handles_labels();
subplot2_legend_labels = ax[1::2][-1].get_legend_handles_labels();

handles = np.append(subplot1_legend_labels[0], subplot2_legend_labels[0]);
labels = np.append(subplot1_legend_labels[1], subplot2_legend_labels[1]);

fig.legend(handles, labels, ncol = labels.size, loc='upper center');

# tight layout takes care of avoiding overlapping of axis labels, etc.
# height is reduced so that there is a free space at the top to contain the 
# figure legend:
fig.tight_layout(rect=(0,0,1,0.95));     

#%% plot section:
    
# for spline smoothing, we wrap the data at the 0/360 degree angle to avoid 
# inconsistensies:
overlap = 20;
angle_vector4spline = np.hstack((angle_vector_deg[-overlap:]-360,
                                 angle_vector_deg,
                                 angle_vector_deg[:overlap]+360));

result_vector4spline = np.hstack((radii[-overlap:],
                                 radii,
                                 radii[:overlap]));

# do the spline interpolation:
radii_vector_spline_interp = UnivariateSpline(angle_vector4spline, result_vector4spline, k = 5, s = 1);
radii_vector_spline = radii_vector_spline_interp(angle_vector_deg);

# plot the final result: 
plt.figure();
plt.subplot(121);
plt.imshow(image_data, cmap="gray");
plt.plot(centroid_x, centroid_y, '.b');
# plot the detected edge:
plt.plot(centroid_x + radii*np.cos(angle_vector_rad),
        centroid_y - radii*np.sin(angle_vector_rad),
        '--b',linewidth=1);

# compare the edge detection methods:
plt.subplot(122);
plt.plot(angle_vector_deg, result_threshold, '.-', label="threshold");
plt.plot(angle_vector4spline, result_vector4spline, '.-', label="cubic fit");
plt.plot(angle_vector_deg, radii_vector_spline, '.-', label="cubic fit + spline");
plt.legend();

#%% show plots (if not run within ipython console)
plt.show()