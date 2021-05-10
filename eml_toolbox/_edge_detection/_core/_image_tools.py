import cv2 as _cv2;
from ._centroid import auto_threshold_borders as _auto_threshold_borders;

#%%
def resize_image(image_data, resize_factor = 0.5):
    """ Resizes the image
    
    Resize the image specified to the size specified by resize_factor

    Parameters
    ----------
    image_data : array_like
        The array containing the image data.
    resize_factor : float, optional
        Defines the factor by which image will be resized in terms of the fraction
        new image size to old image size. The default is 0.5.

    Returns
    -------
    array_like
        array containing the resized image.

    """
    
    # calculate new shape:
    width_new = int(resize_factor*image_data.shape[1]);
    height_new = int(resize_factor*image_data.shape[0]);
    
    # return result:
    return _cv2.resize(image_data,
                       (width_new, height_new),
                       interpolation = _cv2.INTER_AREA);


#%% way faster than imageio!!!
def load_image(image_file):
    """Loads specified image from file
    
    Load the specified image from file using the opencv imread routine and 
    returns the image as an array

    Parameters
    ----------
    image_file : string
        The absolute file path string of the image to load
    
    Returns
    -------
    image_data : array_like
        The array containing the image data
    """
    
    return _cv2.imread(image_file,_cv2.IMREAD_GRAYSCALE);

#%% faster than np.invert()!!!
def invert_image(image_data):
    """Inverts the given image
    
    Inverts the give image data (array) using the opencv bitwise_not routine 
    and returns the inverted image as an array

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    
    Returns
    -------
    image_data_inverted : array_like
        The array containing the inverted image data
    """
    
    return _cv2.bitwise_not(image_data);

# routine to check if image has to be inverted:
def check_image_invert(image_data, border_width = 30):
    """Checks if image has to be inverted
    
    This function checks if the given image is not a surface tension image 
    (bright sample on dark background). The check is done by using the 
    additional return values of the auto_threshold_borders function. If the 
    intensity at the borders of the image is lower than inside the borders 
    then it must be a surface tension image; otherwise, it needs to be inverted. 

    Parameters
    ----------
    image_data : array_like
        The array containing the image data to be checked for inverting
    border_width : int
        The width of the border used in the auto_threshold_borders function
    
    Returns
    -------
    needs_invert : bool
        True/false depending if image needs to be inverted or not before 
        further processing
    """
    
    _, avg_intensity_borders, avg_intensity_inside = \
        _auto_threshold_borders(image_data,border_width = border_width);
    
    # if image borders are darker than the mean image, it's a surface tension
    # image:
    if(avg_intensity_inside > avg_intensity_borders):
        return False;
    # else, it's a shadowgraph image:
    else:
        return True;