""" Provides the coarse edge procedure

This module provides the coarse edge procedures for sub-pixel edge detection. 

"""
# pylint: disable-msg=too-many-locals,too-many-arguments
from . import _core
from .. import _various
import numpy as _np;
import copy as _cp;
from scipy.ndimage.filters import uniform_filter1d as _uniform_filter1d;
from scipy.interpolate import interp1d as _interp1d;
from scipy.interpolate import UnivariateSpline as _UnivariateSpline;
from numpy.polynomial.legendre import legfit as _legendre_fit;
from numpy.polynomial.legendre import legval as _legendre_eval;
from scipy import optimize as _opt;

#%% define default values:
_DEFAULT_EDGE_FILTER_FUNCTION_CALL = [_core.filter_sobel, {"normalize": False}]
_DEFAULT_RESIZE_FACTOR = 1
_DEFAULT_EDGE_THRESHOLD = [100, 1000]
_DEFAULT_EDGE_PIXELS_CUT_OFF_LIMITS = [0.90, 1.10]
_DEFAULT_EDGE_PIXELS_CUT_OFF_ADD = [-1,1]
_DEFAULT_ROLL_PERCENT = 5
_DEFAULT_METHOD = "mvgAvg"
_DEFAULT_METHOD_OPTIONS = {"fit": {"fit_deg": 15},
                          "mvgAvg": {"mvgAvg_width_percent": _DEFAULT_ROLL_PERCENT},
                          "spline": {"mvgAvg_width_percent": _DEFAULT_ROLL_PERCENT,
                                     "spline_width": 0.5},
                          "legendre": {"legendre_deg": 6},
                          "legendre_optimized": {"legendre_deg": 6},
                          "mvgAvg_weighted": {"mvgAvg_width_percent": _DEFAULT_ROLL_PERCENT},
                          }

#%%
def process_image(image_data_array, 
                  angle_vector_rad,
                  auto_threshold_call = [_core.auto_threshold_borders, {"border_width":30} ],
                  threshold = None,
                  centroid = None,
                  centroid_function_call = [_core.centroid, {}],
                  resize_factor = _DEFAULT_RESIZE_FACTOR,
                  edge_filter_function_call = _DEFAULT_EDGE_FILTER_FUNCTION_CALL,
                  edge_threshold = _DEFAULT_EDGE_THRESHOLD,
                  roll_percent = _DEFAULT_ROLL_PERCENT,
                  edge_pixels_cut_off_limits = _DEFAULT_EDGE_PIXELS_CUT_OFF_LIMITS,
                  edge_pixels_cut_off_add = _DEFAULT_EDGE_PIXELS_CUT_OFF_ADD,
                  method = _DEFAULT_METHOD,
                  method_options = None,
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
        DESCRIPTION. The default is [_core.auto_threshold_borders, {"border_width":30} ].
    threshold : TYPE, optional
        DESCRIPTION. The default is None.
    centroid : TYPE, optional
        DESCRIPTION. The default is None.
    centroid_function_call : TYPE, optional
        DESCRIPTION. The default is [_core.centroid, {}].
    resize_factor : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_RESIZE_FACTOR.
    edge_filter_function_call : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_FILTER_FUNCTION_CALL.
    edge_threshold : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_THRESHOLD.
    roll_percent : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_ROLL_PERCENT.
    edge_pixels_cut_off_limits : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_PIXELS_CUT_OFF_LIMITS.
    edge_pixels_cut_off_add : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_PIXELS_CUT_OFF_ADD.
    method : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_METHOD.
    method_options : TYPE, optional
        DESCRIPTION. The default is None.
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
    radius_approx_vector : TYPE
        DESCRIPTION.
    function_parameters : TYPE
        DESCRIPTION.
    TYPE
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
    function_parameters["edge_filter_function_call"][0] = edge_filter_function_call[0].__name__
    function_parameters["centroid_function_call"][0] = centroid_function_call[0].__name__
    
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
    
    # coarse edge detection:
    radius_approx_vector, debug_data = radius_coarse(image_data_array,
                                                     centroid_x,
                                                     centroid_y, 
                                                     angle_vector_rad,
                                                     resize_factor = resize_factor,
                                                     edge_filter_function_call = edge_filter_function_call,
                                                     edge_threshold = edge_threshold,
                                                     roll_percent = roll_percent,
                                                     edge_pixels_cut_off_limits = edge_pixels_cut_off_limits,
                                                     edge_pixels_cut_off_add = edge_pixels_cut_off_add,
                                                     method = method,
                                                     method_options = method_options,
                                                     debug = debug);
    
    # append debug data:
    if debug:
        # add debug data from process_image routine:
        debug_data["threshold"] = threshold;
    
    return centroid_x, centroid_y, pixel_count, radius_approx_vector, function_parameters, debug_data;


#%%
def radius_coarse(image_data,
                  centroid_x,
                  centroid_y,
                  angle_vector_rad, 
                  resize_factor = _DEFAULT_RESIZE_FACTOR,
                  edge_filter_function_call = _DEFAULT_EDGE_FILTER_FUNCTION_CALL,
                  edge_threshold = _DEFAULT_EDGE_THRESHOLD,
                  roll_percent = _DEFAULT_ROLL_PERCENT,
                  edge_pixels_cut_off_limits = _DEFAULT_EDGE_PIXELS_CUT_OFF_LIMITS,
                  edge_pixels_cut_off_add = _DEFAULT_EDGE_PIXELS_CUT_OFF_ADD,
                  method = _DEFAULT_METHOD,
                  method_options = None,
                  debug = False
                  ):
    """
    ToDo

    Parameters
    ----------
    image_data : TYPE
        DESCRIPTION.
    centroid_x : TYPE
        DESCRIPTION.
    centroid_y : TYPE
        DESCRIPTION.
    angle_vector_rad : TYPE
        DESCRIPTION.
    resize_factor : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_RESIZE_FACTOR.
    edge_filter_function_call : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_FILTER_FUNCTION_CALL.
    edge_threshold : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_THRESHOLD.
    roll_percent : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_ROLL_PERCENT.
    edge_pixels_cut_off_limits : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_PIXELS_CUT_OFF_LIMITS.
    edge_pixels_cut_off_add : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_EDGE_PIXELS_CUT_OFF_ADD.
    method : TYPE, optional
        DESCRIPTION. The default is _DEFAULT_METHOD.
    method_options : TYPE, optional
        DESCRIPTION. The default is None.
    debug : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    edge_pixels_radii_coarse : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    if method_options is None:
        method_options = _DEFAULT_METHOD_OPTIONS[method]
    
    if(resize_factor != 1):
        # resize image to speedup coarse edge detection:
        image_data_resized = _core.resize_image(image_data, resize_factor);
    else:
        image_data_resized = image_data;
    
    # run edge detection via filter (e.g. sobel, laplacian):
    image_data_edge = edge_filter_function_call[0](image_data_resized,
                                                   **edge_filter_function_call[1]);
    
    # find the coordinates of the edge pixels by analyzing filtered image and 
    # thresholding:
    row_indices_vector = _np.atleast_2d(_np.arange(0,image_data.shape[0])).T;
    column_indices_vector = _np.arange(image_data.shape[1]);
    
    column_indices, row_indices = _np.broadcast_arrays(column_indices_vector,
                                                         row_indices_vector)
    
    # if any of the array elements contains NaNs, suppress the warning:
    if _np.isnan(image_data_edge).any():
        with _np.errstate(invalid='ignore'):
            edge_pixels_coordinates = _various.where(_np.logical_and(image_data_edge > edge_threshold[0], 
                                                                     image_data_edge < edge_threshold[1]))
    else:
        edge_pixels_coordinates = _various.where(_np.logical_and(image_data_edge > edge_threshold[0], 
                                                                 image_data_edge < edge_threshold[1]))
                
    #edge_pixels_coordinates = _np.where(image_data_edge > edge_threshold);
    
    # starting from centroid, calculate horizontal and vertical distance to 
    # the edge points. Note: edes[0] is the vertical (y) coordinate!
    distance_horizontal = edge_pixels_coordinates[1] - centroid_x*resize_factor;
    distance_vertical = edge_pixels_coordinates[0] - centroid_y*resize_factor;
    
    # calculate radius for each edge pixel and the according angle:
    edge_pixels_radii = _np.sqrt(distance_horizontal**2 + distance_vertical**2);
    # - distance vertical to correspond with image coordinate system!
    edge_pixels_angles = _np.arctan2(-distance_vertical, distance_horizontal);
    
    # map angles from -pi -> pi to angles from 0 -> 2*pi:
    edge_pixels_angles[edge_pixels_angles < 0] = edge_pixels_angles[edge_pixels_angles < 0] + 2*_np.pi;
    
    # sort arrays for further processing:
    sort_index = _np.argsort(edge_pixels_angles);
    edge_pixels_radii_sort = edge_pixels_radii[sort_index];
    edge_pixels_angles_sort = edge_pixels_angles[sort_index];

    # cut off extreme radiis via median:
    edge_pixels_radii_median = _np.median(edge_pixels_radii_sort);
    cut_off_index = _np.logical_or(edge_pixels_radii_sort < edge_pixels_cut_off_limits[0]*edge_pixels_radii_median,
                                   edge_pixels_radii_sort > edge_pixels_cut_off_limits[1]*edge_pixels_radii_median)
    edge_pixels_radii_sort = edge_pixels_radii_sort[~cut_off_index];
    edge_pixels_angles_sort = edge_pixels_angles_sort[~cut_off_index];
    
    # omit "outliers":
    outliers_index_vector = _np.logical_or(edge_pixels_radii_sort < (edge_pixels_radii_sort.mean() - edge_pixels_radii_sort.var() + edge_pixels_cut_off_add[0]),
                                           edge_pixels_radii_sort > (edge_pixels_radii_sort.mean() + edge_pixels_radii_sort.var() + edge_pixels_cut_off_add[1]))
    
    edge_pixels_radii_sort = edge_pixels_radii_sort[~outliers_index_vector]
    edge_pixels_angles_sort = edge_pixels_angles_sort[~outliers_index_vector]
    edge_pixels_values_sort = (image_data_edge[edge_pixels_coordinates])[sort_index];
    
    # add outermost values to the end and the beginning of the array to prevent
    # instabilites for angles near to 0 or 2*pi:
    roll_count = int(roll_percent/100 * edge_pixels_radii_sort.size);
    # edge_pixels_angles_sort = _np.hstack((_np.roll(edge_pixels_angles_sort,roll_count)[0:roll_count] - 2*_np.pi, 
    #                                       edge_pixels_angles_sort, 
    #                                       edge_pixels_angles_sort[0:roll_count] + 2*_np.pi))
    edge_pixels_angles_sort = _various.wrap_angle(edge_pixels_angles_sort, roll_count)
    edge_pixels_radii_sort = _various.wrap(edge_pixels_radii_sort, roll_count)
    edge_pixels_values_sort = _various.wrap(edge_pixels_values_sort, roll_count)
    
    # calculate filter_width where needed:
    filter_width = lambda x: int(x/100* edge_pixels_radii_sort.size)
    
    # initialize debug data variable:
    debug_data = {}
    
    # fit data with polynomial:
    if(method == "fit"): 
        fit_coefficients = _np.polyfit(edge_pixels_angles_sort, 
                                       edge_pixels_radii_sort, 
                                       deg = method_options["fit_deg"]
                                       );
        
        edge_pixels_radii_coarse = _np.polyval(fit_coefficients, angle_vector_rad);
    
    # faster method with only linear interpolation:    
    elif(method == "mvgAvg"):      
        # apply moving average filter:
        edge_pixels_angles_sort_mvgAvg = _uniform_filter1d(edge_pixels_angles_sort, 
                                                           filter_width(method_options["mvgAvg_width_percent"]));
        edge_pixels_radii_sort_mvgAvg = _uniform_filter1d(edge_pixels_radii_sort,
                                                          filter_width(method_options["mvgAvg_width_percent"]));
        # interpolate linearly:
        edge_pixels_radii_coarse = _np.interp(angle_vector_rad,
                                              edge_pixels_angles_sort_mvgAvg,
                                              edge_pixels_radii_sort_mvgAvg)
    
    # spline interpolation:
    elif(method == "spline"):
        # also if spline interpolation is used, first apply a moving average 
        # filter:
        edge_pixels_angles_sort_mvgAvg = _uniform_filter1d(edge_pixels_angles_sort,
                                                           filter_width(method_options["mvgAvg_width_percent"]));
        edge_pixels_radii_sort_mvgAvg = _uniform_filter1d(edge_pixels_radii_sort,
                                                          filter_width(method_options["mvgAvg_width_percent"]));
        
        edge_pixels_radii_interp = _interp1d(edge_pixels_angles_sort_mvgAvg,
                                             edge_pixels_radii_sort_mvgAvg)
        edge_pixels_radii_interp = edge_pixels_radii_interp(angle_vector_rad);
        
        # create function that interpolates the values with slines + smoothing:
        edge_pixels_radii_interp = _UnivariateSpline(angle_vector_rad,
                                                     edge_pixels_radii_interp,
                                                      k = 3,
                                                      s = method_options["spline_width"],
                                                      #s = int(spline_width*(edge_pixels_radii_sort.size \
                                                      #                      - _np.sqrt(2*edge_pixels_radii_sort.size))
                                                      #        ),
                                                      );
        edge_pixels_radii_coarse = edge_pixels_radii_interp(angle_vector_rad);
    
    elif(method == "legendre"):
        edge_pixels_radii_coarse = _radii_fit_legendre(edge_pixels_angles_sort,
                                                       edge_pixels_radii_sort,
                                                       angle_vector_rad,
                                                       method_options["legendre_deg"]);
    
    elif(method == "legendre_optimized"):
        
        selection = _np.logical_and(edge_pixels_angles_sort >= 0, edge_pixels_angles_sort <= 2*_np.pi)
        
        theta_opt = _opt.minimize_scalar(_test_theta,
                                         args=(edge_pixels_angles_sort[selection],
                                               edge_pixels_radii_sort[selection], 
                                               method_options["legendre_deg"]),
                                         method='Bounded',
                                         bounds=(-_np.pi, _np.pi)).x;
        
        coeff_opt = _legendre_fit(_np.cos(edge_pixels_angles_sort[selection] + theta_opt),
                                  edge_pixels_radii_sort[selection],
                                  method_options["legendre_deg"]);
        
        edge_pixels_radii_coarse = _legendre_eval(_np.cos(angle_vector_rad + theta_opt), 
                                                  coeff_opt)
        
        if debug:
            debug_data["theta_opt"] = theta_opt
        
    elif(method == "mvgAvg_weighted"):
        # apply weighted moving average filter:
        edge_pixels_radii_sort_mvgAvg_weighted = _various.mvgAvg_weighted(edge_pixels_radii_sort,
                                                                          edge_pixels_values_sort,
                                                                          filter_width(method_options["mvgAvg_width_percent"]));
        
        # create function that interpolates the values linearly:
        edge_pixels_radii_interp = _interp1d(edge_pixels_angles_sort,
                                             edge_pixels_radii_sort_mvgAvg_weighted);
        
        edge_pixels_radii_coarse = edge_pixels_radii_interp(angle_vector_rad);
        
    else:
        raise(NotImplementedError);
     
    # rescale to the original image size:
    edge_pixels_radii_coarse = 1/resize_factor*edge_pixels_radii_coarse;
    edge_pixels_radii_sort = 1/resize_factor*edge_pixels_radii_sort;
    

    if debug:     
        # compose debug data:
        debug_data["image_data_edge"] = image_data_edge
        debug_data["edge_pixels_coordinates"] = edge_pixels_coordinates
        debug_data["edge_pixels_angles_sort"] = edge_pixels_angles_sort
        debug_data["edge_pixels_radii_sort"] = edge_pixels_radii_sort
        debug_data["edge_pixels_values_sort"] = edge_pixels_values_sort
    
    return edge_pixels_radii_coarse, debug_data;

#%%
def _remap_phi_to_theta(phi_vector_sort):
    """
    ToDo

    Parameters
    ----------
    phi_vector_sort : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # remap the angles from phi (0 -> 2pi) to theta (-pi to pi), since vertical
    # axis symmetry is assumed:
    remap_logical_index_1 = _np.logical_and(phi_vector_sort > _np.pi/2, phi_vector_sort < 3/2*_np.pi);
    remap_logical_index_2 = phi_vector_sort <= _np.pi/2;
    remap_logical_index_3 = phi_vector_sort >= 3/2*_np.pi;
    
    # create copy of original vector:
    theta_vector = _np.copy(phi_vector_sort);
    
    # now start the remapping:
    theta_vector[remap_logical_index_1] = -(phi_vector_sort[remap_logical_index_1] - _np.pi/2);
    theta_vector[remap_logical_index_2] = _np.abs(phi_vector_sort[remap_logical_index_2] - 1/2*_np.pi);
    theta_vector[remap_logical_index_3] = _np.abs(phi_vector_sort[remap_logical_index_3] - 5/2*_np.pi);
    
    return theta_vector;

#%%
def _radius_legendre(coeff, theta):
    """
    ToDo

    Parameters
    ----------
    coeff : TYPE
        DESCRIPTION.
    theta : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return _legendre_eval(_np.cos(theta),coeff);

#%%
def _coeff_legendre(theta_array, radii_array, deg = 6):
    """
    ToDo

    Parameters
    ----------
    theta_array : TYPE
        DESCRIPTION.
    radii_array : TYPE
        DESCRIPTION.
    deg : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return _legendre_fit(_np.cos(theta_array), radii_array, deg);

def _test_theta(diff_theta, theta_array, radii_array, legendre_deg = 6):
    """
    ToDo

    Parameters
    ----------
    diff_theta : TYPE
        DESCRIPTION.
    theta_array : TYPE
        DESCRIPTION.
    radii_array : TYPE
        DESCRIPTION.
    legendre_deg : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return _np.sqrt(_np.sum( (
                                radii_array - _radius_legendre( _coeff_legendre(theta_array + diff_theta, 
                                                                                radii_array,
                                                                                legendre_deg), 
                                                               theta_array + diff_theta) 
                             )**2 )
                    );

#%%
def _radii_fit_legendre(phi_vector_sort, radii_vector_sort, angle_vector_rad, deg = 6):
    """
    ToDo

    Parameters
    ----------
    phi_vector_sort : TYPE
        DESCRIPTION.
    radii_vector_sort : TYPE
        DESCRIPTION.
    angle_vector_rad : TYPE
        DESCRIPTION.
    deg : TYPE, optional
        DESCRIPTION. The default is 6.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    # remap phi to theta
    theta_vector = _remap_phi_to_theta(phi_vector_sort);
    
    # fit with legendre:
    coeff_legendre = _legendre_fit(_np.cos(theta_vector), radii_vector_sort, deg);
    
    # remap angle vector to theta
    angle_vector_rad_theta = _remap_phi_to_theta(angle_vector_rad);
    
    # calculate legendre radii
    radii_legendre = _legendre_eval(_np.cos(angle_vector_rad_theta), coeff_legendre);
    
    # return value:
    return radii_legendre;

# EOF