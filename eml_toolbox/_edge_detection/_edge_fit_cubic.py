""" Provides the cubic fit procedure

This module provides the cubic fit procedure for sub-pixel edge detection. 

"""
# pylint: disable-msg=too-many-locals,too-many-arguments
import numpy as _np
import copy as _cp
from . import _core
from .. import _various
from scipy.ndimage.filters import uniform_filter1d as _uniform_filter1d;

def process_image(image_data_array, 
                  angle_vector_rad,
                  edge_filter_call = [_core.filter_kirsch, {"normalize": True}],
                  auto_threshold_call = [_core.auto_threshold_borders, {"border_width": 30}],
                  threshold = None,
                  centroid = None,
                  centroid_function_call = [_core.centroid,{}],
                  fine_factor = 5,
                  angle_filter_call = [_uniform_filter1d, {}],
                  angle_filter_window_size_rad = _np.deg2rad(10),
                  edge_filter_threshold = 0.20,
                  edge_filter_add_tuple = (-0,+0),
                  force_fit_values_tuple = ((1,5),(0.9,0)),
                  debug = False
                  ):
    """ Processes the image and determines the centroid and radii
    
    For the image specified by image_data_array, the centroid and the radii for 
    all search angles specified in angle_vector_rad are determined. 

    Parameters
    ----------
    image_data_array : array_like
        The array containing the image data.
    angle_vector_rad : array_like
        Vector containing the search angles in radian.
    edge_filter_call : list, optional
        List containing the edge filter function at position 0 and possible keyword 
        arguments at position 1. The default is [_core.filter_kirsch, {"normalize": True}].
    auto_threshold_call : list, optional
        List containing the auto threshold function at position 0 and possible keyword 
        arguments at position 1. The default is [_core.auto_threshold_borders, {"border_width": 30}].
    threshold : int, optional
        Intensity threshold value used to determine the centroid. If None,
        the threshold is automatically determined. The default is None.
    centroid_function_call : list, optional
        List containing the centroid function at position 0 and possible keyword 
        arguments at position 1. The default is [_core.centroid,{}].
    fine_factor : int, optional
        The radial search vectors standard step size of 1 px is divided by this 
        factor to get a better resolution. The default is 5.
    angle_filter_call : list, optional
        List containing the angle filter function at position 0 and possible keyword 
        arguments at position 1. The default is [_uniform_filter1d, {}].
    angle_filter_window_size_rad : float, optional
        Window size for the moving average filter over neighboring angles in radian. 
        The default is _np.deg2rad(10).
    edge_filter_threshold : int, optional
        The (relative) edge filter threshold used to determine where the intensity 
        profile is fitted with the cubic polynomial. The default is 0.20.
    edge_filter_add_tuple : tuple, optional
        Allows additional pixels from the radial profile to be included to the
        cubic fit procedure. The default is (-0,+0).
    force_fit_values_tuple : tuple, optional
        Describes how the first and last element of the intensity profile to be fit
        are modified to force the fit to converge. The first number is multiplied with the 
        intensity value and the second number is added to this result. Therefore, the
        data is unmodified for ((1,0),(1,0)). The default is ((1,5),(0.9,0)).
    debug : bool, optional
        Whether debug data should be collected or not. The default is False.

    Returns
    -------
    centroid_x : float
        The horizontal centroid coordinate in px
    centroid_y : float
        The vertical centroid coordinate in px
    pixel_count : int
        Number of pixels above the threshold.
    radii_vector : array_like
        Vector containing the radii for all search directions.
    function_parameters : dict
        Dictionary containing all function parameters supplied
    debug_data : dict
        Dictionary containing the collected debug data.

    """
       
    # save function parameters used for later reference:
    function_parameters = _cp.deepcopy(locals())
    del function_parameters["image_data_array"]
    del function_parameters["angle_vector_rad"]
    function_parameters["angle_filter_call"][0] = angle_filter_call[0].__name__
    function_parameters["auto_threshold_call"][0] = auto_threshold_call[0].__name__
    function_parameters["centroid_function_call"][0] = centroid_function_call[0].__name__
    function_parameters["edge_filter_call"][0] = edge_filter_call[0].__name__
    
    # check datatype and if it's uint8, convert it to int16 to avoid unexpected
    # behaviour (e.g. negative values):
    if image_data_array.dtype is _np.dtype("uint8"):   
        image_data_array = _various.change_dtype(image_data_array, dtype = _np.int16)
    
    #%% apply edge filter to image:
    edge_data_array = edge_filter_call[0](image_data_array,
                                          **edge_filter_call[1]);
    
    #%% centroid part:
    # only calculate centroid if not already given (e.g. when used as standalone
    # version):
    if centroid is None:
        # centroid part:
        if threshold is None:
            # automatically determine threshold and check consistency of all variants: 
            #threshold, _, _ = auto_threshold_function(image_data_array);
            threshold, _, _ = auto_threshold_call[0](image_data_array, 
                                                     **auto_threshold_call[1]);
    
        # get centroid
        centroid_x, centroid_y, pixel_count = centroid_function_call[0](image_data_array,
                                                                        threshold,
                                                                        **centroid_function_call[1]);
    else:
        centroid_x = centroid["centroid_x"]
        centroid_y = centroid["centroid_y"]
        pixel_count = centroid["pixel_count"]
        threshold = centroid["threshold"]
    
    #%% preparation:    
    # go half the image diagonal to the outside:
    radius = _np.round(1/2*_np.sqrt(image_data_array.shape[0]**2 + image_data_array.shape[1]**2));
    radius = radius.astype(int);
    
    # generate (subpixel) coordinates of the line: 
    # resolution of radius line is 1px, thus coordinates of arbitrary line points are between pixels (subpixel coordinates)
    steps = radius*fine_factor + 1;
    
    # create radius vector:
    radius_vector = _np.linspace(0, radius, steps);
    
    # get line coordinates:
    coordinates_x_array, coordinates_y_array = _core.line_coordinates(image_data_array,
                                                                      centroid_x,
                                                                      centroid_y,
                                                                      angle_vector_rad,
                                                                      step_division = fine_factor);
    
    # create x,y pairs of all intensity profile interpolation points:
    coordinates_array = _np.vstack((coordinates_x_array.ravel(),
                                    coordinates_y_array.ravel()));
    
    # get intensity values at all x,y coordinates
    intensity_profiles_array = _core.map_coordinates(image_data_array,
                                                     coordinates_array,
                                                     cval = _np.NaN);
    
    edge_profiles_array = _core.map_coordinates(edge_data_array, 
                                                coordinates_array,
                                                cval = 0);
    
    # reshape to original shape:
    intensity_profiles_array = _np.reshape(intensity_profiles_array, 
                                           coordinates_x_array.shape);
    edge_profiles_array = _np.reshape(edge_profiles_array, 
                                      coordinates_x_array.shape);
    
    #%% now include neighboring pixels via moving average over angles:
    # calculate number of angles for moving average filter:
    angle_filter_window_size = int(angle_filter_window_size_rad / 
                                   (angle_vector_rad[1]-angle_vector_rad[0]));
    
    
    if angle_filter_window_size > 1:    
        intensity_profiles_array = angle_filter_call[0](intensity_profiles_array,
                                                        angle_filter_window_size,
                                                        axis = -1,
                                                        mode="wrap",
                                                        **angle_filter_call[1]);
        
        edge_profiles_array = angle_filter_call[0](edge_profiles_array,
                                                   angle_filter_window_size,
                                                   axis = -1,
                                                   mode="wrap",
                                                   **angle_filter_call[1]);
    
    #%% hand over to fit function:
    radii_vector, debug_data = intensity_profiles_fit_cubic(angle_vector_rad, 
                                                            fine_factor, 
                                                            radius_vector,
                                                            intensity_profiles_array,
                                                            edge_profiles_array,
                                                            threshold,
                                                            edge_filter_threshold,
                                                            edge_filter_add_tuple,
                                                            force_fit_values_tuple,
                                                            debug);
    
    # append debug data:
    if debug:
        debug_data["threshold"] = threshold
        debug_data["radius_vector"] = radius_vector
        debug_data["intensity_profiles_array"] = intensity_profiles_array
        debug_data["edge_profiles_array"] = edge_profiles_array
    
    return centroid_x, centroid_y, pixel_count, radii_vector, function_parameters, debug_data;

#%%
def intensity_profiles_fit_cubic(angle_vector_rad,
                                 fine_factor,
                                 radius_vector,
                                 intensity_profiles_array,
                                 edge_profiles_array,
                                 threshold,
                                 edge_filter_threshold,
                                 edge_filter_add_tuple,
                                 force_fit_values_tuples,
                                 debug
                                 ):
    """ Fits the intensity profiles with a cubic polynomial
    
    Fits the intensity profiles specified by intensity_profiles_array with a
    cubic polynomial to find the sub-pixel edge detection

    Parameters
    ----------
    angle_vector_rad : array_like
        Vector containing the search angles in radian.
    fine_factor : int, optional
        The radial search vectors standard step size of 1 px is divided by this 
        factor to get a better resolution. The default is 5.
    radius_vector : array_like
        Vector containing the standard radial search positions in steps of 1 px.
    intensity_profiles_array : array_like
        Array containing the radial intensity profiles for all search angles at
        the radial positions of radius_vector.
    edge_profiles_array : array_like
        Array containing the radial edge intensity profiles for all search angles at
        the radial positions of radius_vector.
    threshold : int, optional
        Intensity threshold value used to determine the centroid. If None,
        the threshold is automatically determined. The default is None.
    edge_filter_threshold : int, optional
        The (relative) edge filter threshold used to determine where the intensity 
        profile is fitted with the cubic polynomial. The default is 0.20.
    edge_filter_add_tuple : tuple, optional
        Allows additional pixels from the radial profile to be included to the
        cubic fit procedure. The default is (-0,+0).
    force_fit_values_tuple : tuple, optional
        Describes how the first and last element of the intensity profile to be fit
        are modified to force the fit to converge. The first number is multiplied with the 
        intensity value and the second number is added to this result. Therefore, the
        data is unmodified for ((1,0),(1,0)). The default is ((1,5),(0.9,0)).
    debug : bool, optional
        Whether debug data should be collected or not. The default is False.

    Returns
    -------
    radii_vector : array_like
        Vector containing the radii for all search directions.
    debug_data : dict
        Dictionary containing the collected debug data.

    """
    
    # preallocate
    radii_vector = _np.zeros(angle_vector_rad.size)
    radii_threshold_vector = _np.zeros(angle_vector_rad.size)
    
    radius_vector_fit_all_list, intensity_vector_fit_all_list, fit_coeff_all_list = [], [], []
    
    #%% loop through different angles:
    for i in range(0, angle_vector_rad.size, 1):
            
        # get current intensity profile:
        intensity_profile_vector = intensity_profiles_array[:, i]
        
        # get approximate edge range of sample by using threshold value:
        approx_edge_threshold_index = (_np.where(intensity_profile_vector < threshold)[0])[0]
        approx_edge_threshold = radius_vector[approx_edge_threshold_index]
        
        # if sample does NOT touch image boundaries:
        if intensity_profile_vector[int(approx_edge_threshold_index*1.1)] != _np.NaN:
            
            # get edge_filter profile:
            edge_profile_vector = edge_profiles_array[:, i]
            
            # normalize:
            edge_profile_vector = edge_profile_vector/_np.nanmax(_np.abs(edge_profile_vector))
            
            # find approximate edge position by analyzing edge profile:
            approx_edge_index = _np.argmax(edge_profile_vector)
            approx_edge_index_in = _np.where(
                edge_profile_vector[approx_edge_index::-1] < edge_filter_threshold)
            
            if _np.size(approx_edge_index_in) > 0:
                approx_edge_index_in = approx_edge_index_in[0][0]
            else:
                approx_edge_index_in = 0
                    
                
            approx_edge_index_out = _np.where(
                edge_profile_vector[approx_edge_index:] < edge_filter_threshold)
            
            if _np.size(approx_edge_index_out) > 0:
                approx_edge_index_out = approx_edge_index_out[0][0]
            else:
                approx_edge_index_out = _np.size(radius_vector)-1
            
            approx_edge_index_in = approx_edge_index - approx_edge_index_in
            approx_edge_index_out = approx_edge_index + approx_edge_index_out
            
            radius_fit_min_index = approx_edge_index_in + edge_filter_add_tuple[0]*fine_factor
            radius_fit_max_index = approx_edge_index_out + edge_filter_add_tuple[1]*fine_factor
            
            if radius_fit_min_index < 0:
                radius_fit_min_index = 0
                
            if radius_fit_max_index >= _np.size(radius_vector):
                radius_fit_max_index = _np.size(radius_vector) - 1
    
            radius_vector_fit = radius_vector[radius_fit_min_index:
                                              radius_fit_max_index + 1]
                
            intensity_vector_fit = _np.copy(intensity_profile_vector[radius_fit_min_index:
                                                                     radius_fit_max_index + 1])
            
            # force polyfit to find "correct" polynomial with point of deflection
            # at edge:
            intensity_vector_fit[0] = intensity_vector_fit[0]*force_fit_values_tuples[1][0] + \
                                        force_fit_values_tuples[1][1]
            intensity_vector_fit[-1] = intensity_vector_fit[-1]*force_fit_values_tuples[0][0] + \
                                        force_fit_values_tuples[0][1]
            
            # fit data with cubic polynomial:
            fit_coeff = _np.polyfit(radius_vector_fit - _np.mean(radius_vector_fit),
                                    intensity_vector_fit - _np.mean(intensity_vector_fit),
                                    deg=3)
                                                                                   
            # get point of inflection:
            radius = -1/3*fit_coeff[1]/fit_coeff[0] + _np.mean(radius_vector_fit)
        
        else:
            print('WARNING: Sample touches image boundary at angle %i°!' % i)
            radius = _np.NaN
            
        radii_vector[i] = radius
        radii_threshold_vector[i] = approx_edge_threshold
        
        radius_vector_fit_all_list.append(radius_vector_fit)
        intensity_vector_fit_all_list.append(intensity_vector_fit)
        fit_coeff_all_list.append(fit_coeff)
    
    debug_data = {}
    if debug: 
        debug_data["radii_threshold_vector"] = radii_threshold_vector
        debug_data["radii_vector_fit_all_list"] = radius_vector_fit_all_list
        debug_data["intensity_vector_fit_all_list"] = intensity_vector_fit_all_list
        debug_data["fit_coeff_all_list"] = fit_coeff_all_list

    return radii_vector, debug_data
