import numpy as np;
cimport numpy as cnp;
cimport cython;

#%% auto-threshold cython version (for documentation, please see the pure 
# numpy version)
@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int auto_threshold_cython(cnp.uint8_t[:, :] image_data_view_2d):
    """please see docstring of auto_threshold function for more information!"""
    
    cdef int image_height = image_data_view_2d.shape[0];
    cdef int image_width = image_data_view_2d.shape[1];
    cdef int not_black_count = 0, not_black_intensity_sum = 0;
    cdef int pixel_count = 0;
    cdef int coordinate_x, coordinate_y, threshold;
      
    for coordinate_x in range(image_width):
        for coordinate_y in range(image_height):
            if image_data_view_2d[coordinate_y, coordinate_x] > 0:
                not_black_count += 1;
                not_black_intensity_sum += image_data_view_2d[coordinate_y, coordinate_x];
    
    threshold = int(0.5*not_black_intensity_sum/not_black_count);
    
    return threshold;

#%% centroid-cython version (for documentation, please see the pure numpy 
# version)
@cython.cdivision(True)
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef (double, double, int) centroid_cython(cnp.uint8_t[:, :] image_data_view_2d, int threshold):
    """please see docstring of centroid function for more information!"""
    
    cdef int image_height = image_data_view_2d.shape[0];
    cdef int image_width = image_data_view_2d.shape[1];
    cdef int pixel_count = 0;
    cdef double centroid_x = 0, centroid_y = 0;
    cdef int coordinate_x, coordinate_y;
    
    for coordinate_x in range(image_width):
        for coordinate_y in range(image_height):
            if image_data_view_2d[coordinate_y, coordinate_x] > threshold:
                centroid_x += coordinate_x;
                centroid_y += coordinate_y;
                pixel_count += 1;
        
    return (centroid_x/pixel_count, centroid_y/pixel_count, pixel_count);