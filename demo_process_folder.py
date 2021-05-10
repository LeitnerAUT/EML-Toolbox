import eml_toolbox as eml;
import numpy as np;
import glob;
from matplotlib import pyplot as plt;
from joblib import Parallel, delayed

#%% first close everything:
plt.close("all");

angle_vector_deg_step_size = 1;
threshold = 200

file_comment = "edge_cubic"

angle_vector_deg = np.arange(0, 360, step = 1);
angle_vector_rad = np.deg2rad(angle_vector_deg);

# get content of directory:
folder_name = "TU Graz side view (density)"
image_list = glob.glob("demo_images/%s/*.tif*" % folder_name);

# for testing, reduce number of images to process:
#image_list = image_list[0:10];

def load_and_process_image(image_file, full_output = False):
    image_data = eml.ed.load_image(image_file)
    
    check_invert = eml.ed.check_image_invert(image_data)
    if(check_invert):
        image_data = eml.ed.invert_image(image_data); # faster than np.invert()!!!
    
        
    centroid_x, centroid_y, pixel_count, radii, function_parameters, debug_data = \
        eml.ed.edge_fit_cubic.process_image(image_data,
                                            angle_vector_rad,
                                            threshold = threshold)
        
    # centroid_x, centroid_y, pixel_count, radii, function_parameters, debug_data = \
    #     eml.ed.edge_coarse.process_image(image_data,
    #                                       angle_vector_rad,
    #                                       method = "fit")
    
    if full_output:
        return centroid_x, centroid_y, pixel_count, radii, function_parameters, debug_data
    else:
        return np.hstack((centroid_x, centroid_y, pixel_count, radii))


def process_folder(image_list):
    
    # obtain function parameters
    _, _, _, _, function_parameters, _ = load_and_process_image(image_list[0], full_output = True)
    
    result_folder = np.array(
                            Parallel(n_jobs = -1, verbose = 1)
                            (delayed(load_and_process_image)(image_file)
                              for image_file in image_list)
                            );
    
    return result_folder, function_parameters

# process the folder:
result_folder, function_parameters = process_folder(image_list);

#%% save results:
# prepare comments:
comments = "".join("#" + keys + ":" + str(values) + "\n" for keys, values in function_parameters.items());
# save to txt file:
np.savetxt("demo_data/%s_radii_%s_%1.1fdeg.txt" % (folder_name, file_comment ,angle_vector_deg_step_size), 
            result_folder, 
            delimiter = '\t',
            fmt = '%1.3f',
            header = "c_x\t c_y\t npix\t r0...r359",
            comments = comments);

#%% visually check results:
plt.figure(figsize=(12,9));

plt.subplot(221)
plt.plot(result_folder[:,0])
plt.title("centroid x")
plt.xlabel("frame no.")
plt.ylabel("x coordinate / px")

plt.subplot(222)
plt.plot(result_folder[:,1])
plt.title("centroid y")
plt.xlabel("frame no.")
plt.ylabel("y coordinate / px")

plt.subplot(212)
plt.plot(result_folder[:,3:])
plt.title("Radii")
plt.xlabel("frame no.")
plt.ylabel("radius / px")

plt.tight_layout()
plt.savefig("demo_data/%s_%s_plot1.png" % (folder_name, file_comment))

#%%
plt.figure(figsize=(12,9));
# image_no = 5
# result_folder = np.atleast_2d(result_folder[image_no,:])
plt.imshow(eml.ed.load_image(image_list[0]), cmap="gray")
plt.plot(np.tile(np.atleast_2d(result_folder[:,0]).T, (1,angle_vector_rad.size)) + result_folder[:,3:]*np.cos(angle_vector_rad),
         np.tile(np.atleast_2d(result_folder[:,1]).T, (1,angle_vector_rad.size)) - result_folder[:,3:]*np.sin(angle_vector_rad),
         ".",
         markersize=1)
plt.axis("equal")
plt.tight_layout()
plt.savefig("demo_data/%s_%s_plot2.png" % (folder_name, file_comment))

#%% show plots (if not run within ipython console)
plt.show()