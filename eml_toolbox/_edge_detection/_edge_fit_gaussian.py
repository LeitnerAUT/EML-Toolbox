""" Provides the gaussian fit procedure

This module provides the gaussian fit procedure for sub-pixel edge detection. 

"""
# pylint: disable-msg=too-many-locals,too-many-arguments
from . import _core;
from .. import _various;
from . import _edge_coarse as _edge_coarse
import numpy as _np
import copy as _cp
from scipy.ndimage.filters import uniform_filter1d as _uniform_filter1d;

_DEFAULT_METHOD = "standard"
_DEFAULT_METHOD_OPTIONS = {"coarse_edge_first": {"resize_factor" : 0.5,
                                                 "coarse_edge_filter_threshold": [100, _np.inf],
                                                 "radius_approx_interval": 10},
                           "edge_image": {"edge_filter_call": [_core.filter_sobel, {}],
                                          "edge_filter_threshold": 100},
                           "standard": {}
                           }

#%%
def process_image(image_data_array,
                  angle_vector_rad,
                  auto_threshold_call = [_core.auto_threshold_borders, {"border_width": 30}],
                  threshold = None,
                  centroid = None,
                  centroid_function_call = [_core.centroid,{}],
                  method = _DEFAULT_METHOD,
                  method_options = None,
                  step_division = 1,
                  angle_filter_window_size_rad = None,
                  sigma = 3,
                  threshold_grad = 0.1,
                  edge_threshold_search_limit = [0.9,1.1],
                  debug = False
                  ):
    """
    ToDo

    Parameters
    ----------
    image_data_array : TYPE
        DESCRIPTION.
    angle_vector_rad : TYPE
        DESCRIPTION.
    auto_threshold_call : TYPE, optional
        DESCRIPTION. The default is [_core.auto_threshold_borders, {"border_width": 30}].
    threshold : TYPE, optional
        DESCRIPTION. The default is None.
    centroid_function_call : TYPE, optional
        DESCRIPTION. The default is [_core.centroid,{}].
    method : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_METHOD.
    method_options : TYPE, optional
        DESCRIPTION. The default is None.
    step_division : TYPE, optional
        DESCRIPTION. The default is 1.
    angle_filter_window_size_rad : TYPE, optional
        DESCRIPTION. The default is None.
    sigma : TYPE, optional
        DESCRIPTION. The default is 3.
    threshold_grad : TYPE, optional
        DESCRIPTION. The default is 0.1.
    edge_threshold_search_limit : TYPE, optional
        DESCRIPTION. The default is [0.9,1.1].
    debug : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    centroid_x : TYPE
        DESCRIPTION.
    centroid_y : TYPE
        DESCRIPTION.
    pixel_count : TYPE
        DESCRIPTION.
    radii_vector : TYPE
        DESCRIPTION.
    function_parameters : TYPE
        DESCRIPTION.
    debug_data : TYPE
        DESCRIPTION.

    """
    
    # load methods default options if not supplied:
    if method_options is None:
        method_options = _DEFAULT_METHOD_OPTIONS[method]
    
    # save function parameters used for later reference:
    function_parameters = _cp.deepcopy(locals())
    del function_parameters["image_data_array"]
    del function_parameters["angle_vector_rad"]
    function_parameters["auto_threshold_call"][0] = auto_threshold_call[0].__name__
    function_parameters["centroid_function_call"][0] = centroid_function_call[0].__name__
    
    # initialize:
    debug_data = {}
    
    # if no angle averaging is given (default), the default value is set to the
    # step size of angle_vector_rad:
    if angle_filter_window_size_rad is None:
        angle_filter_window_size_rad = angle_vector_rad[1]-angle_vector_rad[0];
    
    # calculate number of items for angle averaging:
    angle_filter_window_size = int(angle_filter_window_size_rad / 
                               (angle_vector_rad[1]-angle_vector_rad[0]));
    
    
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
      
    # check datatype and if it's uint8, convert it to int16 to avoid unexpected
    # behaviour (e.g. negative values):
    if image_data_array.dtype is _np.dtype("uint8"):   
        image_data_array = _various.change_dtype(image_data_array, dtype = _np.int16)
    
    # if only edge image should be used:
    if method == "edge_image":
        
        # replace image data array by edge image:        
        image_data_array = method_options["edge_filter_call"][0](image_data_array,
                                                                 **method_options["edge_filter_call"][1]);
        
        # replace threshold by edge_filter_threshold:
        threshold = method_options["edge_filter_threshold"]
        
        function_parameters["method_options"]["edge_filter_call"][0] = \
            function_parameters["method_options"]["edge_filter_call"][0].__name__
        
        calc_grad = False
    
    else:
        calc_grad = True
    
    #%% if coarse_edge detection is used:
    if method == "coarse_edge_first":
        
        # get coarse edge from (resized) image:
        _, _, _, radius_approx_vector, function_parameters["edge_coarse"], \
            debug_data["edge_coarse"] = _edge_coarse.process_image(image_data_array,
                                                                   angle_vector_rad,
                                                                   centroid = {"centroid_x": centroid_x,
                                                                               "centroid_y": centroid_y,
                                                                               "pixel_count": pixel_count,
                                                                               "threshold": threshold
                                                                       },
                                                                   resize_factor = method_options["resize_factor"],
                                                                   edge_threshold = method_options["coarse_edge_filter_threshold"],
                                                                   method = "mvgAvg",
                                                                   debug = debug);
        
        # if angle averaging is set, make sure that radius_approx_vector starts at
        # the same radius for all angles:
        if angle_filter_window_size > 1:
            
            radius_approx_interval = 3*int(_np.round(_np.max(radius_approx_vector) - _np.min(radius_approx_vector)))
            
            radius_approx_vector = _np.ones_like(radius_approx_vector)*\
                1/2*(_np.max(radius_approx_vector) + _np.min(radius_approx_vector))       
        
        # get coordinates along the radial lines:
        coordinates_x, coordinates_y = _core.line_coordinates(image_data_array,
                                                              centroid_x,
                                                              centroid_y, 
                                                              angle_vector_rad,
                                                              step_division = step_division,
                                                              radius_approx = radius_approx_vector,
                                                              radius_approx_interval = method_options["radius_approx_interval"]);
    else:
        
        # get coordinates along the radial lines:
        coordinates_x, coordinates_y = _core.line_coordinates(image_data_array, 
                                                              centroid_x,
                                                              centroid_y, 
                                                              angle_vector_rad,
                                                              step_division = step_division);
    
    #%% continue:
    # create x,y pairs of all intensity profile interpolation points:
    coordinates_array = _np.vstack((coordinates_x.ravel(),coordinates_y.ravel()));
        
        
    # get intensity values at all x,y coordinates
    intensity_profiles = _core.map_coordinates(image_data_array,
                                               coordinates_array,
                                               mode = "nearest");
        
    # reshape to original shape:
    intensity_profiles = _np.reshape(intensity_profiles, coordinates_x.shape);
    
    # now include neighboring pixels via moving average over angles:
    if angle_filter_window_size > 1:
        intensity_profiles = _uniform_filter1d(intensity_profiles,
                                               size = angle_filter_window_size,
                                               axis = -1,
                                               mode="wrap");
    
    #%%                                                               
    if not method == "coarse_edge_first":
        # calculate radius:
        image_width = image_data_array.shape[1];
        image_height = image_data_array.shape[0];
        radius = step_division*int(_np.round(0.5 * _np.sqrt(image_width ** 2 + image_height ** 2)));
        
        # cut off values clearly not dedicated to the object (black):
        mask_sample = intensity_profiles > threshold
        
        # find approximate edge by searching for first occurence of pixel above the 
        # treshold from outside to inside:
        r_threshold_inverse = _np.argmax(_np.flip(mask_sample, axis = 0), axis = 0);
        r_threshold = radius - r_threshold_inverse;
        r_min_vec = _np.round(r_threshold*edge_threshold_search_limit[0]).astype(_np.int);
        r_max_vec = _np.round(r_threshold*edge_threshold_search_limit[1]).astype(_np.int);
        
        # define boundaries where gradient is analyzed:
        r_min = _np.min(r_min_vec);
        r_max = _np.max(r_max_vec);
        
        # get edges:
        intensity_profiles_tmp = intensity_profiles[r_min:r_max,:];
        
    else:
        intensity_profiles_tmp = intensity_profiles;
        
    #%%    
    # get edges:
    mu_vector, sigma_vector, debug_data["calc_edge"] = _core.calc_edge(intensity_profiles_tmp,
                                                                       sigma = sigma,
                                                                       calc_grad = calc_grad,
                                                                       threshold_grad = threshold_grad);
    
    #%%
    if method == "coarse_edge_first":    
        radii_vector = 1/step_division*mu_vector + radius_approx_vector - method_options["radius_approx_interval"];
        
        if debug:
            debug_data["radius_approx_vector"] = radius_approx_vector;
            if angle_filter_window_size > 1:
                debug_data["calc_edge"]["r_vector"] = 1/step_division*(_np.arange(0, intensity_profiles.shape[0])) \
                                                        + radius_approx_vector[0] \
                                                        - method_options["radius_approx_interval"];
            else:
                debug_data["calc_edge"]["r_vector"] = 1/step_division*(_np.tile(_np.atleast_2d(_np.arange(0, intensity_profiles.shape[0])).T,
                                                                                (1,intensity_profiles.shape[1]))) \
                                                                       + _np.tile(radius_approx_vector,
                                                                                (intensity_profiles.shape[0],1)) \
                                                                       - method_options["radius_approx_interval"]
    else:
        radii_vector = 1/step_division*(mu_vector + r_min);
        
        if debug:
            debug_data["calc_edge"]["r_min"] = r_min
            debug_data["calc_edge"]["r_vector"] = 1/step_division*(_np.arange(0, intensity_profiles_tmp.shape[0]) + r_min);
    
    #%%
    if debug:    
        debug_data["threshold"] = threshold;
        debug_data["intensity_profiles"] = intensity_profiles
        debug_data["mu_vector"] = mu_vector
        debug_data["sigma_vector"] = sigma_vector
    
    return centroid_x, centroid_y, pixel_count, radii_vector, function_parameters, debug_data
