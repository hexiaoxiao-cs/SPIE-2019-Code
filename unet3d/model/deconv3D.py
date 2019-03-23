import numpy as np
import warnings
from keras.engine import Layer, InputSpec
from keras.layers.convolutional import Conv3D
from keras.legacy import interfaces
from keras.utils import conv_utils
from unet3d.model import backend_updated as K

class Deconvolution3D(Conv3D):
    @interfaces.legacy_conv3d_support
    def __init__(self, filters, kernel_size, 
                 padding='valid', strides=(1, 1, 1),
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 data_format=None, activation=None, 
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 use_bias=True, **kwargs):
        # if data_format is None:
            # data_format = K.image_data_format()
        # if padding not in {'valid', 'same', 'full'}:
            # raise Exception('Invalid border mode for Deconvolution3D:', padding)
        super(Deconvolution3D, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                              data_format=data_format, activation=activation, use_bias=use_bias,
                                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                              activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
                                              bias_constraint=bias_constraint, **kwargs)
        self.input_spec = InputSpec(ndim=5)
        return
    ##
    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError('Inputs should have rank '+str(5)+'; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs ' 'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)
        self.kernel = self.add_weight(kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight((self.filters,), initializer=self.bias_initializer,
                                        name='bias', regularizer=self.bias_regularizer, constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True
        return
    ##
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
        else:
            c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.strides
        output_shape[c_axis] = self.filters
        output_shape[d_axis] = conv_utils.deconv_length(output_shape[d_axis], stride_d, kernel_d, self.padding)
        output_shape[h_axis] = conv_utils.deconv_length(output_shape[h_axis], stride_h, kernel_h, self.padding)
        output_shape[w_axis] = conv_utils.deconv_length(output_shape[w_axis], stride_w, kernel_w, self.padding)
        #print('compute_outptu_shape', output_shape)
        return tuple(output_shape)
    ##
    def call(self, inputs):
        input_shape = K.shape(inputs)
        output_shape = [0]*5
        output_shape[0] = input_shape[0]
        if self.data_format == 'channels_first':
            c_axis, d_axis, h_axis, w_axis = 1, 2, 3, 4
        else:
            c_axis, d_axis, h_axis, w_axis = 4, 1, 2, 3
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.strides
        output_shape[c_axis] = self.filters
        output_shape[d_axis] = conv_utils.deconv_length(input_shape[d_axis], stride_d, kernel_d, self.padding)
        output_shape[h_axis] = conv_utils.deconv_length(input_shape[h_axis], stride_h, kernel_h, self.padding)
        output_shape[w_axis] = conv_utils.deconv_length(input_shape[w_axis], stride_w, kernel_w, self.padding)
        #print('call',output_shape)
        outputs = K.deconv3d(inputs, self.kernel, output_shape, strides=self.strides,
                             padding=self.padding, data_format=self.data_format)
        if self.bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
    ##
    def get_config(self):
        # config = {'output_shape': self.output_shape_}
        config = super(Deconvolution3D, self).get_config()
        config.pop('dilation_rate')
        return config
        # return dict(list(base_config.items()) + list(config.items()))