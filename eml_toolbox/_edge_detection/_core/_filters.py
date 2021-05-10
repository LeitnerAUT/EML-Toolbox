import numpy as _np;
import cv2 as _cv2;


#%% Laplacian:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
def filter_laplacian(image_data, kernel_size = 3, normalize = False, relative_norm = False):
    """Laplacian edge filter
    
    Applies a Laplacian edge filter to the given image data
    

    Parameters
    ----------
    image_data : array_like
        The array containing the image data
    kernel_size : int, optional
        The kernel size of the laplacian filter. The default is 3.
    normalize : bool, optional
        If set to True, the values are normalized to uint8 (0 to 255). The default is False.

    Returns
    -------
    array_like
        array containing the edge filter image.

    """
    laplacian = _cv2.Laplacian(image_data, -1, ksize = kernel_size);
    
    # normalize to 8bit grayscale:
    if normalize:
        laplacian = _cv2.normalize(_np.abs(laplacian), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
        
    if relative_norm:
        laplacian = laplacian/_np.max(_np.abs(laplacian))
    
    return laplacian;

#%% Sobel
# https://stackoverflow.com/questions/51167768/sobel-edge-detection-using-opencv
def filter_sobel(image_data, kernel_size = -1, normalize = False, relative_norm = False):
    """Sobel edge filter
    
    Applies a Sobel edge filter to the given image data:
    

    Parameters
    ----------
    image_data : array_lile
        The array containing the image data
    kernel_size : TYPE, optional
        The kernel size of the laplacian filter. The default is -1.
    normalize : bool, optional
        If set to True, the values are normalized to uint8 (0 to 255). The default is False.

    Returns
    -------
    array_like
        Array containing the edge filter image.

    """    
    # apply sobel filter in horizontal direction:
    sobel_x = _cv2.Sobel(image_data, -1, 1, 0, ksize = kernel_size)
    # apply sobel filter in vertical direction:
    sobel_y = _cv2.Sobel(image_data, -1, 0, 1, ksize = kernel_size)
    
    # build absolute values, rescales already to uint8:
    # abs_grad_x = _cv2.convertScaleAbs(sobel_x)
    # abs_grad_y = _cv2.convertScaleAbs(sobel_y)
    
    # build absolute values:
    _np.abs(sobel_x, out = sobel_x)
    _np.abs(sobel_y, out = sobel_y)
    
    # combine the edge information from horizontal and vertical runs:
    sobel_xy = _cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    # normalize to 8bit grayscale:
    if normalize:
        sobel_xy = _cv2.normalize(sobel_xy, None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    
    if relative_norm:
        sobel_xy = sobel_xy/_np.max(sobel_xy)
    
    return sobel_xy;


#%% Kirsch Operator Code
# taken from: https://stackoverflow.com/a/38445233
def filter_kirsch(image_data, normalize = False, relative_norm = False):
    """Kirsch edge filter
    
    Applies a Kirsch edge filter to the given image data:

    Parameters
    ----------
    image_data : array_like
        Array containing the image data
    normalize : bool, optional
        If set to True, the values are normalized to uint8 (0 to 255). The default is False.

    Returns
    -------
    edge_data : array_like
        Array containing the edge filter image.

    """
    # dtype=_np.float32
    # NOTE: initializing the kernels outside the function does NOT pay off 
    # regarding computation time!
    kernelG1 = _np.array([[ 5,  5,  5],
                         [-3,  0, -3],
                         [-3, -3, -3]], dtype=image_data.dtype)
    kernelG2 = _np.array([[ 5,  5, -3],
                         [ 5,  0, -3],
                         [-3, -3, -3]], dtype=image_data.dtype)
    kernelG3 = _np.array([[ 5, -3, -3],
                         [ 5,  0, -3],
                         [ 5, -3, -3]], dtype=image_data.dtype)
    kernelG4 = _np.array([[-3, -3, -3],
                         [ 5,  0, -3],
                         [ 5,  5, -3]], dtype=image_data.dtype)
    kernelG5 = _np.array([[-3, -3, -3],
                         [-3,  0, -3],
                         [ 5,  5,  5]], dtype=image_data.dtype)
    kernelG6 = _np.array([[-3, -3, -3],
                         [-3,  0,  5],
                         [-3,  5,  5]], dtype=image_data.dtype)
    kernelG7 = _np.array([[-3, -3,  5],
                         [-3,  0,  5],
                         [-3, -3,  5]], dtype=image_data.dtype)
    kernelG8 = _np.array([[-3,  5,  5],
                         [-3,  0,  5],
                         [-3, -3, -3]], dtype=image_data.dtype)

    # if normalize:
    #     g1 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG1), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g2 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG2), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g3 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG3), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g4 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG4), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g5 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG5), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g6 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG6), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g7 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG7), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    #     g8 = _cv2.normalize(_cv2.filter2D(image_data, _cv2.CV_32F, kernelG8), None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    
    # g1 = _cv2.filter2D(image_data, _cv2.CV_32F, kernelG1)
    g1 = _cv2.filter2D(image_data, -1, kernelG1)
    g2 = _cv2.filter2D(image_data, -1, kernelG2)
    g3 = _cv2.filter2D(image_data, -1, kernelG3)
    g4 = _cv2.filter2D(image_data, -1, kernelG4)
    g5 = _cv2.filter2D(image_data, -1, kernelG5)
    g6 = _cv2.filter2D(image_data, -1, kernelG6)
    g7 = _cv2.filter2D(image_data, -1, kernelG7)
    g8 = _cv2.filter2D(image_data, -1, kernelG8)
    
    edge_data = _cv2.max(
        g1, _cv2.max(
            g2, _cv2.max(
                g3, _cv2.max(
                    g4, _cv2.max(
                        g5, _cv2.max(
                            g6, _cv2.max(
                                g7, g8
                            )
                        )
                    )
                )
            )
        )
    )
    
    # normalize to 8bit grayscale:
    if normalize:
        edge_data = _cv2.normalize(edge_data, None, 0, 255, _cv2.NORM_MINMAX, _cv2.CV_8UC1)
    
    if relative_norm:
        edge_data = edge_data/_np.max(edge_data)
    
    return edge_data;

# EOF