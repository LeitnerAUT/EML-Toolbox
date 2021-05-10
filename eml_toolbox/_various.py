import numpy as _np;

#%%
def nan_compare(compare_function, input_array, compare_value):
    """ Only applies the compare function to non-NaN values
    ToDo...
    

    Parameters
    ----------
    compare_function : TYPE
        DESCRIPTION.
    input_array : TYPE
        DESCRIPTION.
    compare_value : TYPE
        DESCRIPTION.

    Returns
    -------
    mask_not_nan : TYPE
        DESCRIPTION.

    """
    mask_not_nan = ~_np.isnan(input_array)
    mask_not_nan[mask_not_nan] = compare_function(input_array[mask_not_nan] , compare_value)
    
    return mask_not_nan

#%%
def where(condition_array):
    """ Custom where function to replace numpy.where for speed enhancement
    
    This function was written to replace the numpy.where function due to performance
    issues. Like numpy.where, it returns the row- and column indices of the 
    array entries of the input condition_array which are True. 

    Parameters
    ----------
    condition_array : array_like
        The logical array (condition array, e.g. array == True)

    Returns
    -------
    array_like
        Vector containing the row indices where condition_array is True.
    array_like
        Vector containing the column indices where condition_array is True.

    """
    # first create the row and column vectors:
    row_indices_vector = _np.atleast_2d(_np.arange(0,condition_array.shape[0])).T;
    column_indices_vector = _np.arange(condition_array.shape[1]);
    
    # span 2d arrays from them, similar to meshgrid, but faster:
    column_indices, row_indices = _np.broadcast_arrays(column_indices_vector,
                                                       row_indices_vector)
    
    # get the row and column indices by logical indexing:
    return (row_indices[condition_array],
            column_indices[condition_array])

#%%
def wrap_angle(array, wrap_count):
    """ Wraps angle vector
    
    Helper-function that wraps an angle vector containing angles in radian by 
    the number of elements specified by wrap count. The last wrap_count elements in 
    array are stacked to the beginning of the array and lowered by 2*pi and the
    first wrap_count elements in array are stacked to the end and raised by 2*pi.
    

    Parameters
    ----------
    array : array_like
        Vector containing the sorted angles in radian.
    wrap_count : int
        Number of elements that will be wrapped.

    Returns
    -------
    array_like
        Wrapped angle vector.

    """
    return _np.hstack((array[-wrap_count:] - 2*_np.pi, 
                       array, 
                       array[0:wrap_count] + 2*_np.pi))

#%%
def wrap(array, wrap_count):
    """ Wraps vector
    
    Helper-function that wraps a vector containing by the number of elements 
    specified by wrap count. The last wrap_count elements in array are stacked 
    to the beginning of the array and the first wrap_count elements in array 
    are stacked to the end.
    

    Parameters
    ----------
    array : array_like
        Vector to be wrapped.
    wrap_count : int
        Number of elements that will be wrapped.

    Returns
    -------
    array_like
        Wrapped vector.

    """
    return _np.hstack((array[-wrap_count:], 
                       array, 
                       array[0:wrap_count]))

#%%
def change_dtype(data, dtype = _np.int16):
    """ Changes data dtype
    
    Changes the data type of the input data to the specified data type

    Parameters
    ----------
    data : array_like
        The data whose datatype should be changed.
    dtype : dtype, optional
        The data type to which the input should be changed. The default is _np.int16.

    Returns
    -------
    array_like
        Data of new data type.

    """
    # if data has already the correct data type, do nothing
    if data.dtype == dtype:
        return data
    # else, change the data type:
    else:
        return data.astype(dtype);
#%%
def stop():
    """ Stops the execution of the script
    
    Small helper function that stops the execution of a python script at the 
    position where it is called.

    Returns
    -------
    None.

    """
    raise(SystemExit);

#%%
def mvgAvg_weighted(data, weights, window_size = 10):
    """ Calculates weighted moving average of data
    
    Try-out implementation to perform a weighted moving average on the input data

    Parameters
    ----------
    data : array_like
        Vector containing the data on which the weighted moving average is 
        calculated.
    weights : array_like
        Vector containing the weights for each element in data.
    window_size : int, optional
        Defines the window size for the weighted moving average filtering. 
        The default is 10.

    Returns
    -------
    array_like
        The weighted moving average filtered data.

    """
    
    ind = _np.arange(0, data.shape[0]);
    
    result = _np.zeros_like(data);
    
    for i in ind:
        roll_index = i - int(window_size/2);
        
        result[i] = _np.average(_np.roll(data,-roll_index)[0:window_size], 
                               weights=_np.roll(weights,-roll_index)[0:window_size]);
   
    return result;