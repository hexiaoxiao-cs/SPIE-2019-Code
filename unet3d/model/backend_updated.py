import tensorflow as tf
from keras.backend import *

"""
## copy the implementations from previous keras version
"""
##
def _pre_preprocess_conv3d_input(x, data_format):
    """
    # Transpose and cast the input before the conv3d.
    # Arguments
        x: input tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    if dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if data_format == 'channels_first':
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    return x
##
def _pre_preprocess_conv3d_kernel(kernel, data_format):
    """
    # Transpose and cast the kernel before the conv3d.
    # Arguments
        kernel: kernel tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    if dtype(kernel) == 'float64':
        kernel = tf.cast(kernel, 'float32')
    if data_format == 'channels_first':
        kernel = tf.transpose(kernel, (2, 3, 4, 1, 0))
    return kernel
##
def _pre_postprocess_conv3d_output(x, data_format):
    """
    # Transpose and cast the output from conv3d if needed.
    # Arguments
        x: A tensor.
        data_format: string, `"channels_last"` or `"channels_first"`.
    # Returns
        A tensor.
    """
    if data_format == 'channels_first':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    if floatx() == 'float64':
        x = tf.cast(x, 'float64')
    return x
##
def _pre_preprocess_padding(padding):
    """
    # Convert keras' padding to tensorflow's padding.
    # Arguments
        padding: string, `"same"` or `"valid"`.
    # Returns
        a string, `"SAME"` or `"VALID"`.
    # Raises
        ValueError: if `padding` is invalid.
    """
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding:', padding)
    return padding
##
def _preprocess_deconv3d_output_shape(x, shape, data_format):
    if data_format == 'channels_first':
        shape = (shape[0], shape[2], shape[3], shape[4], shape[1])
    if shape[0] is None:
        shape = (tf.shape(x)[0], ) + tuple(shape[1:])
        shape = tf.stack(list(shape))
    return shape

"""
## deconv3d
"""
def deconv3d(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None, filter_shape=None):
    """
    # 3D deconvolution (i.e. transposed convolution).
    # Arguments
        x: input tensor.
        kernel: kernel tensor.
        output_shape: 1D int tensor for the output shape.
        strides: strides tuple.
        padding: string, "same" or "valid".
        data_format: "tf" or "th".
            Whether to use Theano or TensorFlow dimension ordering
            for inputs/kernels/ouputs.
    """
    # print '*** Deconv3d: \n\tX:{0} \n\tkernel:{1} \n\tstride:{2} \n\tfilter_shape:{3} \n\toutput_shape:{4}'.format(int_shape(x), int_shape(kernel), strides, filter_shape, output_shape)
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))
    if isinstance(output_shape, (tuple, list)):
        output_shape = tf.stack(output_shape)
    x = _pre_preprocess_conv3d_input(x, data_format)
    output_shape = _preprocess_deconv3d_output_shape(x, output_shape, data_format)
    # kernel = _pre_preprocess_conv3d_kernel(kernel, data_format)
    # kernel = tf.transpose(kernel, (0, 1, 2, 4, 3))
    strides = (1,) + strides + (1,)
    padding = _pre_preprocess_padding(padding)
    # print '*** Deconv3d: \n\tkernel:{0} \n\tfilter_shape:{1} '.format(int_shape(kernel), filter_shape)
    # print output_shape
    x = tf.nn.conv3d_transpose(x, kernel, output_shape, strides, padding)
    return _pre_postprocess_conv3d_output(x, data_format)