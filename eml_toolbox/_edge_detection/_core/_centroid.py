import numpy as _np;

#%% 
def auto_threshold_borders(image_data, border_width = 30, weight = 2/3):
    """Determines the threshold value for the given image data using the 
    borders method
    
    The threshold value for the centroid detection is determined by using the
    borders method. The image borders defined by the width in pixel are 
    analyzed for their mean intensity. Also for the inside of the borders, the 
    mean intensity is calculated. The threshold is then defined as the mean 
    intensity in the borders region plus the difference between the two
    intensities times the weight-factor, e.g. 1/2

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    border_width : int, optional
        The width of the border region analyzed in pixels (default is 30)
    weight : float, optional
        The weight-factor (between 0 and 1) shifting the threshold to the 
        value of the border region or to the value inside the borders (default 
        is 2/3)
    
    Returns
    -------
    threshold
        value of the threshold
    avg_intensity_borders
        mean intensity in the image's border region
    avg_intensity_inside
        mean intensity inside the image's border region
    """

    # create mask to select only the border pixels:
    mask = _np.ones(image_data.shape, dtype=_np.bool);
    mask[border_width:-border_width,border_width:-border_width] = False;
    
    # calculate average intensity of this border region:
    avg_intensity_borders = _np.mean(image_data[mask]);
    
    # calculate average intensity inside the border:
    avg_intensity_inside = _np.mean(image_data[~mask]);
    
    # threshold is mean value of avg intensity of whole image and boundary 
    # region
    threshold = _np.int(_np.round(avg_intensity_borders + weight*_np.abs(avg_intensity_borders - avg_intensity_inside)));
    
    return threshold, _np.int(_np.round(avg_intensity_borders)), _np.int(_np.round(avg_intensity_inside)); 

#%% 
def auto_threshold(image_data):
    """Determines the threshold value for the given image data using the half-
    intensity method
    
    The threshold value for the centroid detection is determined by using the
    half-intensity method. The threshold is simply calculated by 1/2 of the 
    mean intensity of all non-black pixels. 
    This method should not be used in case of shadowgraph-images where 
    background intensity is usually not zero!

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    
    Returns
    -------
    threshold
        value of the threshold
    """
      
    # get all non-black pixels:
    mask_not_black = image_data > 0;
    
    # get intensity of all non-black pixels (1d-vector):
    pixels_not_black_intensities = image_data[mask_not_black];
    
    # calculate auto-threshold:
    threshold = _np.int(0.5*_np.mean(pixels_not_black_intensities));
    
    return threshold;

#%% 
def centroid(image_data, threshold): 
    """Calculates the centroid for all pixels above the threshold
    
    The centroid of all pixels in the image that have a larger intensity than 
    the threshold is calculated.

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    threshold : int
        The treshold value
    
    Returns
    -------
    centroid_x
        the coordinate of the centroid in horizontal direction
    centroid_y
        the coordinate of the centroid in vertical direction
    pixel_count
        the number of pixels that have an intensity greater than the threshold
        value
    """
        
    # get pixels above threshold:
    # if any of the array elements contains NaNs, suppress the warning:
    if _np.isnan(image_data).any():
        with _np.errstate(invalid='ignore'):
            mask_above_threshold = image_data > threshold
    else:
        mask_above_threshold = image_data > threshold;

    # count pixels above threshold:    
    pixel_count = _np.sum(mask_above_threshold);        
        
    # get size of image:
    image_height, image_width = image_data.shape;
          
    # create vectors of x and y coordinates for the given image shape:
    coordinates_x = _np.arange(0,image_width);
    coordinates_y = _np.arange(0,image_height);
    
    # calculate centroid in x or y direction by building the matrix product of 
    # the x or y axis sum of the (logical) array of all pixels above the 
    # threshold with the vector of x or y direction:
    centroid_x = _np.sum(mask_above_threshold, axis=0).dot(coordinates_x.T)/pixel_count;
    centroid_y = _np.sum(mask_above_threshold, axis=1).dot(coordinates_y.T)/pixel_count;
    
    # return the centroid coordinates, the pixel count of pixels above the 
    # threshold and the threshold itself:
    return centroid_x, centroid_y, pixel_count;   


#%% no performance gain if auto-threshold is included into main centroid function!
def centroid_auto_threshold_borders(image_data, threshold=-1, border_width=30, weight=2/3, centroid_function=centroid):
    """Calculates the centroid of the sample in the image by automatically 
    determining the threshold using the borders method
    
    If the threshold value is set to -1, then the threshold for the centroid 
    detection is automatically determined by the borders method (please see 
    docstring of the auto_threshold_borders function for more details). 
    Subsequently, using the obtained threshold, the centroid is calculated 
    (please see the centroid_cython docstring for more information).

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    threshold : int, optional
        The treshold value to use for the centroid detection. If -1, the 
        automatic threshold detection is used. (default is -1)
    border_width : int, optional
        The width of the border region analyzed in pixels (default is 30)
    weight : float, optional
        The weight-factor (between 0 and 1) shifting the threshold to the 
        value of the border region or to the value inside the borders (default 
        is 2/3)
    centroid_function : function, optional
        Function used to calculate the centroid. By default, the built-in pure
        numpy centroid function is used. To enhance performance, the cython-version
        can be used by specifying "centroid_cython" as the centroid_funciton (if
        a compiler is installed). The default is centroid.
    
    Returns
    -------
    centroid_x
        the coordinate of the centroid in horizontal direction
    centroid_y
        the coordinate of the centroid in vertical direction
    pixel_count
        the number of pixels that have an intensity greater than the threshold
        value
    """
    
    if(threshold < 0):
        threshold, _, _ = auto_threshold_borders(image_data, border_width = border_width, weight = weight);
            
    return centroid_function(image_data, threshold);

#%% no performance gain if auto-threshold is included into main centroid function!
def centroid_auto_threshold(image_data, threshold=-1, auto_threshold_function=auto_threshold, centroid_function=centroid):
    """Calculates the centroid of the sample in the image by automatically 
    determining the threshold using the half-intensity method
    
    If the threshold value is set to -1, then the threshold for the centroid 
    detection is automatically determined by the half-intensity method (please 
    see the docstring of the auto_threshold function for more details). 
    Subsequently, using the obtained threshold, the centroid is calculated 
    (please see the centroid_cython docstring for more information).

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    threshold : int
        The treshold value to use for the centroid detection. If -1, the 
        automatic threshold detection is used. (default is -1)
    auto_threshold_function : function, optional
        Function used to calculate the threshold. By default, the built-in pure
        numpy auto_threshold function is used. To enhance performance, the cython-version
        can be used by specifying "auto_threshold_cython" as the auto_threshold_funciton 
        (if a compiler is installed). The default is auto_threshold.
    centroid_function : function, optional
        Function used to calculate the centroid. By default, the built-in pure
        numpy centroid function is used. To enhance performance, the cython-version
        can be used by specifying "centroid_cython" as the centroid_funciton (if
        a compiler is installed). The default is centroid.
    
    Returns
    -------
    centroid_x
        the coordinate of the centroid in horizontal direction
    centroid_y
        the coordinate of the centroid in vertical direction
    pixel_count
        the number of pixels that have an intensity greater than the threshold
        value
    """    
    if(threshold < 0):
        threshold = auto_threshold_function(image_data);
            
    return centroid_function(image_data, threshold);

# EOF