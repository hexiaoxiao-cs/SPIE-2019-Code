# python buildin
import sys
import os
from os import listdir
from os.path import isfile, join
from os.path import split, splitext, basename
import numpy as np
import time
from scipy import ndimage
from scipy import misc
# keras-tensorflow
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, add, Conv3D, MaxPooling3D, Dropout, Lambda, Dense, Reshape, Flatten, GlobalAveragePooling1D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from .deconv3D import Deconvolution3D as Deconv3D
from keras.layers.normalization import BatchNormalization
from keras import regularizers

"""
## global setting
"""
if K.image_data_format() == 'channels_first':  # TF dimension ordering in this code
    global_channel_axis = 1
else:
    global_channel_axis = -1
global_smooth = 0.00001 # 1.0

"""
## global setting: PReLU's shared_axes
"""
global_pReLU_shared_axes = [1,2,3]

"""
## dice.
## TODO: check the gradient == 2.0 * ( (y_true * union) - 2.0*y_pred*(intersection+global_smooth) ) / ((union+global_smooth) ** 2) ?
"""
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f = tf.div(tf.add(1.0, y_pred_f), 2.0)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + global_smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + global_smooth)

"""
## dice loss.
"""
def dice_coef_loss(y_true, y_pred):
    return tf.subtract(1.0, dice_coef(y_true, y_pred))

"""
## basic resNet block.
"""
# 1-conv block of resNet
def get_1conv_block_resNet(in_layer, n_feature_map, tl_filter_size, kernel_regularizer):
    conv_mid_1 = Conv3D(n_feature_map, tl_filter_size,
                        activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(in_layer)
    conv_mid_1_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_mid_1)
    add_mid = add([in_layer, conv_mid_1_b])
    out_layer = PReLU(shared_axes=global_pReLU_shared_axes)(add_mid)
    return out_layer
# 2-conv block of resNet
def get_2conv_block_resNet(in_layer, n_feature_map, tl_filter_size, kernel_regularizer):
    conv_mid_1 = Conv3D(n_feature_map, tl_filter_size,
                        activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(in_layer)
    conv_mid_1_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_mid_1)
    conv_mid_1_b_relu = PReLU(shared_axes=global_pReLU_shared_axes)(conv_mid_1_b)
    conv_mid_2 = Conv3D(n_feature_map, tl_filter_size,
                        activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(conv_mid_1_b_relu)
    conv_mid_2_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_mid_2)
    add_mid = add([in_layer, conv_mid_2_b])
    out_layer = PReLU(shared_axes=global_pReLU_shared_axes)(add_mid)
    return out_layer
# 3-conv block of resNet
def get_3conv_block_resNet(in_layer, n_feature_map, tl_filter_size, kernel_regularizer):
    conv_mid_1 = Conv3D(n_feature_map, tl_filter_size,
                        activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(in_layer)
    conv_mid_1_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_mid_1)
    conv_mid_1_b_relu = PReLU(shared_axes=global_pReLU_shared_axes)(conv_mid_1_b)
    conv_mid_2 = Conv3D(n_feature_map, tl_filter_size,
                        activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(conv_mid_1_b_relu)
    conv_mid_2_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_mid_2)
    conv_mid_2_b_relu = PReLU(shared_axes=global_pReLU_shared_axes)(conv_mid_2_b)
    conv_mid_3 = Conv3D(n_feature_map, tl_filter_size,
                        activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(conv_mid_2_b_relu)
    conv_mid_3_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_mid_3)
    add_mid = add([in_layer, conv_mid_3_b])
    out_layer = PReLU(shared_axes=global_pReLU_shared_axes)(add_mid)
    return out_layer
#
def change_num_featmap_1x1x1conv(in_layer, n_new_feature_map, kernel_regularizer):
    temp_layer = Conv3D(n_new_feature_map, (1, 1, 1), activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                        kernel_regularizer=kernel_regularizer)(in_layer)
    temp_layer_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(temp_layer)
    temp_layer_b_relu = PReLU(shared_axes=global_pReLU_shared_axes)(temp_layer_b)
    return temp_layer_b_relu

"""
## generator.
"""
# 1branches + 2pooling + resNet_2conv_2conv_3conv
def get_generator(im_shape, nchannel=(1,), num_class=2, weight_decay=0.0005):
    """
    ## prepare inputs
    """
    # im shape
    if global_channel_axis == 1:
        im_shape = list(nchannel) + list(im_shape)
    else:
        im_shape = list(im_shape) + list(nchannel)
    # l2 regularizer
    kernel_regularizer = regularizers.l2(weight_decay)
    # get input
    inputs = Input(shape=im_shape, name='main_input')

    """
    ## before downsampling
    """
    conv_0_before_dsampling = Conv3D(4, (3, 3, 3),
                                     activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                                     kernel_regularizer=kernel_regularizer)(inputs)
    conv_0_before_dsampling_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(conv_0_before_dsampling)
    conv_0_before_dsampling_b_relu = PReLU(shared_axes=global_pReLU_shared_axes)(conv_0_before_dsampling_b)

    """
    ## downsampling (shared branch)
    """
    ## 0 level
    # 2-conv resNet
    d0b = get_2conv_block_resNet(conv_0_before_dsampling_b_relu, 4, (3, 3, 3), kernel_regularizer)
    print("d0b: ", d0b.get_shape())
    ## 1 level
    # conv2-pooling
    d1a_t = Conv3D(8, (2, 2, 2), strides=(2, 2, 2), use_bias=False, kernel_regularizer=kernel_regularizer)(d0b)
    d1a = PReLU(shared_axes=global_pReLU_shared_axes)(d1a_t)
    print("d1a: ", d1a.get_shape())
    # 2-conv resNet
    d1b = get_2conv_block_resNet(d1a, 8, (3, 3, 3), kernel_regularizer)
    print("d1b: ", d1b.get_shape())
    ## 2 level
    # conv2-pooling
    d2a_t = Conv3D(16, (2, 2, 2), strides=(2, 2, 2), use_bias=False, kernel_regularizer=kernel_regularizer)(d1b)
    d2a = PReLU(shared_axes=global_pReLU_shared_axes)(d2a_t)
    print("d2a: ", d2a.get_shape())
    # 3-conv resNet
    d2b = get_3conv_block_resNet(d2a, 16, (3, 3, 3), kernel_regularizer)
    print("d2b: ", d2b.get_shape())

    """
    ## upsampling (classification branch)
    """
    ## up c: 1 level
    # upooling
    c1a_t = Deconv3D(8, (2, 2, 2), strides=(2, 2, 2), activation=None, padding='valid',
                     kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(d2b)
    c1a = PReLU(shared_axes=global_pReLU_shared_axes)(c1a_t)
    print("c1a: ", c1a.get_shape())
    # concat: d1b + c1a
    concat_d1b_c1a_t = concatenate([d1b, c1a], axis=global_channel_axis)
    concat_d1b_c1a = change_num_featmap_1x1x1conv(concat_d1b_c1a_t, 8, kernel_regularizer)
    c1b = get_2conv_block_resNet(concat_d1b_c1a, 8, (3, 3, 3), kernel_regularizer)
    print("c1b: ", c1b.get_shape())
    ## up c: 0 level
    # upooling
    c0a_t = Deconv3D(4, (2, 2, 2), strides=(2, 2, 2), activation=None, padding='valid',
                     kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(c1b)
    c0a = PReLU(shared_axes=global_pReLU_shared_axes)(c0a_t)
    print("c0a: ", c0a.get_shape())
    # concat: d0b + c0a
    concat_d0b_c0a_t = concatenate([d0b, c0a], axis=global_channel_axis)
    concat_d0b_c0a = change_num_featmap_1x1x1conv(concat_d0b_c0a_t, 4, kernel_regularizer)
    c0b = get_2conv_block_resNet(concat_d0b_c0a, 4, (3, 3, 3), kernel_regularizer)
    print("c0b: ", c0b.get_shape())
    ## fully connected: classification
    c0c_t = Conv3D(4, (3, 3, 3), activation=None, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=kernel_regularizer)(c0b)
    c0c_t_b = BatchNormalization(axis=global_channel_axis, momentum=0.95)(c0c_t)
    c0c_t_b_rl = PReLU(shared_axes=global_pReLU_shared_axes)(c0c_t_b)
    c0c = Dropout(0.2)(c0c_t_b_rl)
    print("c0c: ", c0c.get_shape())
    c0d = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=kernel_regularizer, name='classification_output')(c0c)
    #c0d = Conv3D(num_class, (1, 1, 1), activation='softmax', padding='same', kernel_initializer='he_normal',
    #             kernel_regularizer=kernel_regularizer, name='classification_output')(c0c)
    print("c0d: ", c0d.get_shape())

    """
    ## finish model
    """
    model = Model(inputs=inputs, outputs=c0d)
    return model