import numpy as _np;
from scipy.ndimage import map_coordinates as _map_coordinates;
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d;

#%%
def line_coordinates(image_data, centroid_x, centroid_y, angle_vector_rad, \
                     step_division = 1, radius_approx = None, radius_approx_interval = 20):
    """Get the coordinates along each radial line of each specified angle
    
    For all angles specified in angle_vector_rad, the coordinates (x,y) along 
    a radial line, starting from the centroid coordinates, are calculated with
    a stepsize of 1 px. The result is provided within two vectors, one for each 
    coordinate axis, which is used in the further process for interpolating the
    intensity values at the given positions.
    If a vector containing the approximate radii for all angles is already known,
    it can be specified using the radius_approx variable. Then, the radial lines
    will span from radius_approx +- 1/2 radius_approx_interval.

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    centroid_x : float
        The horizontal coordinate of the centroid in pixels
    centroid_y : float
        The vertical coordinate of the centroid in pixels
    angle_vector_rad : array_like (1D)
        vector containing all angles in radian for which the radial line 
        coordinates are calculated
    step_division : float, optional
        The step size (which defaults to 1 px) is divded by the value of 
        step_division (default is 1)
    radius_approx : array_like, optional
        Vector containing the approximate radius for each radial direction specified
        in angle_vector_rad. The default is None
    radius_approx_interval : int, optional
        Defines the interval of the approximate radii positions (in px) so that the 
        radial profile coordinates will start at the approximate radii +- 1/2 times
        the interval specified. The default is 20.
    
    Returns
    -------
    coordinates_x_array
        array containing all x-coordinates of all radial lines for all angles
    coordinates_y_array
        array containing all y-coordinates of all radial lines for all angles
    radius
        radius of the radial lines (half of the image diagonal)
    """    

    # get size of image:
    image_width = image_data.shape[1];
    image_height = image_data.shape[0];
    
    # for further processing, angle_vector_rad needs to be a row vector:
    # first ensure its 2-dimensional:
    # if(angle_vector_rad.ndim < 2):
    #     angle_vector_rad = _np.atleast_2d(angle_vector_rad);
    # # second if it's a column vector, transpose:
    # elif(angle_vector_rad.shape[0] > angle_vector_rad.shape[1]):
    #     angle_vector_rad = angle_vector_rad.T();
    
    if(radius_approx is None):
        # source coordinates are centroid coordinates:
        coordinates_source = _np.array((centroid_x,centroid_y));
        
        # calculate default radius (1/2 of the image diagonal):   
        radius = int(_np.round(0.5 * _np.sqrt(image_width ** 2 + image_height ** 2)));
        
        coordinates_source_array = _np.meshgrid(angle_vector_rad, coordinates_source)[1];
        # calculate destination coordinates:
        # cm_y - sin, since indexing of image starts from top left:
        coordinates_destination_array = coordinates_source_array + _np.array((radius*_np.cos(angle_vector_rad), -radius*_np.sin(angle_vector_rad)));
        
        # create x and y vectors for interpolation beforehand:
        # radius + 1 since number of elements for vector is radius + 1
        coordinates_x_array = _np.linspace(coordinates_source[0], coordinates_destination_array[0,:], step_division*radius + 1);
        coordinates_y_array = _np.linspace(coordinates_source[1], coordinates_destination_array[1,:], step_division*radius + 1);
    
    else:
        coordinates_source_x = centroid_x + (radius_approx - radius_approx_interval)*_np.cos(angle_vector_rad);
        coordinates_source_y = centroid_y - (radius_approx - radius_approx_interval)*_np.sin(angle_vector_rad);  
        
        coordinates_dest_x = centroid_x + (radius_approx + radius_approx_interval)*_np.cos(angle_vector_rad);
        coordinates_dest_y = centroid_y - (radius_approx + radius_approx_interval)*_np.sin(angle_vector_rad);
        
        coordinates_x_array = _np.linspace(coordinates_source_x, coordinates_dest_x, step_division*2*radius_approx_interval + 1, axis = 0);
        coordinates_y_array = _np.linspace(coordinates_source_y, coordinates_dest_y, step_division*2*radius_approx_interval + 1, axis = 0);
    
    return coordinates_x_array, coordinates_y_array;

#%%
def map_coordinates(image_data, coordinates_array, order = 1, mode = "constant", cval = 0):
    """returns the intensity values at the given coordinates
    
    For each coordinate in the image specified (image_data) in the 
    coordinates_array, the intensity value is determined. The function returns 
    an array containing all intensity values for each coordinate pair (x,y) in
    the coordinates array.
    Please see the documentation of scipy.ndimage.map_coordinates for more 
    information on the optional parameters

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    coordinates_array : array_like
        array with pairs of coordinates (x, y)
    order : int, optional
        order of the interpolation if coordinate is between pixels. The value 
        must be between 0 and 5. 0 corresponds to next neighbor interpolation,
        1 gives a linear interpolation between neighboring pixels (default is 1)
    mode : string, optional
        determines, how values outside the image array are handled. "constant"
        gives a constant value defined by the parameter cval (default is 
        "constant")
    cval : int, optional
        value to be used if mode is set to "constant" (default is 0)
    
    Returns
    -------
    intensities_vector
        vector containing the pixel intensites of the image at the given 
        coordinates
    """
    
    # image data must be transposed to give correct results, please see 
    # https://stackoverflow.com/a/23846484:
    return _map_coordinates( _np.transpose(image_data), 
                            coordinates_array, 
                            order = order, 
                            mode = mode,
                            cval = cval);

#%%
def calc_edge(data, sigma = 3, calc_grad = True, threshold_grad = 0.1):
    """Calculate the edge for the given intensity profile
    
    For each intensity profile specified in the input array (data), the edge is
    determined using a gaussian fit procedure to the first derivative of the
    intensity profile. The optional parameter sigma is used to smooth the data 
    using a gaussian filter before the first derivative is built.

    Parameters
    ----------
    data : array_like
        Array containing the intensity profiles
    sigma : int, optional
        width of the gaussian fit filter (default is 3)
    calc_grad = True : bool, optional
        Defines, whether the gradient should be calculated for the radial intensity
        profiles before further calculation or not. Must be set to False, if
        the radial intensity profiles stem from an edge filter image. The default is True.
    threshold_grad: float, optional
        in order to get a reasonable edge, the profile of the first derivative
        needs to be cut off at values lower than threshold_grad. (default is 
        0.1)
    
    Returns
    -------
    mu_vector
        vector containing the position of the edge along the intensity profile
    sigma_vector
        vector containing the sigmas of the fitted gaussian function to the 
        profile of the first derivative of the intensity profile
    """
    
    # first smooth data:
    data_smooth = _gaussian_filter1d(data, sigma = sigma, axis = 0);
    
    if calc_grad:
        # build 1st derivative:
        data_smooth_grad = _np.gradient(data_smooth, axis = 0);
        
        # cut out areas with wrong gradient, e.g. dark --> bright:
        data_smooth_grad[data_smooth_grad > 0] = 1e-16;
        
        # now build absolute value:
        data_smooth_grad = _np.abs(data_smooth_grad);
    
    else:
        data_smooth_grad = data_smooth;
    
    # remove noise outside of edge:
    # calculate maximum gradient and apply gradient threshold for each angle 
    # individually
    max_grad_vector = _np.atleast_2d(_np.max(data_smooth_grad, axis = 0));
    max_grad_array = _np.repeat(max_grad_vector, data_smooth_grad.shape[0], axis = 0);
    data_smooth_grad[data_smooth_grad < max_grad_array*threshold_grad] = 0;
    
    # inspired by 
    # https://scipy-cookbook.readthedocs.io/items/FittingData.html#Fitting-gaussian-shaped-data
    x_vector = _np.arange(data_smooth_grad.shape[0]);
    mu_vector = _np.dot(data_smooth_grad.T, x_vector)/_np.sum(data_smooth_grad, axis = 0);
    sigma_vector = _np.sqrt(_np.abs(_np.sum(_np.subtract.outer(x_vector, mu_vector)**2 * data_smooth_grad, axis = 0)) / _np.sum(data_smooth_grad, axis = 0))
    
    debug_data = {}
    debug_data["data"] = data
    debug_data["data_smooth"] = data_smooth
    debug_data["data_smooth_grad"] = data_smooth_grad
    debug_data["x_vector"] = x_vector
    
    return mu_vector, sigma_vector, debug_data;
