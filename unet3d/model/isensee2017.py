from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D,Conv3D, Concatenate,Softmax
from keras.engine import Model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
from keras import losses
from keras.utils.np_utils import to_categorical
from .unet import create_convolution_block, concatenate
from ..metrics import dice_coefficient_loss,weighted_dice_coefficient_loss
from keras.activations import softmax

create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(input_shape=(144,144,80,3), n_base_filters=16, depth=3, dropout_rate=0.3,
                      n_segmentation_levels=1, n_labels=3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=losses.categorical_crossentropy, activation_name="softmax"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()

    '''layer1'''
    in_conv = create_convolution_block(current_layer, 16)
    context_output_layer = create_context_module(in_conv, 16, dropout_rate=dropout_rate)
    print(context_output_layer.shape)
    summation_layer = concatenate([in_conv, context_output_layer], axis=-1)
    print(summation_layer.shape)
    level_output_layers.append(summation_layer)
    print(len(level_output_layers))
    current_layer = summation_layer
    print(current_layer.shape)
    '''Layer1-END'''

    '''Layer2-Start'''
    in_conv = create_convolution_block(current_layer, 32, strides=(2, 2, 2))
    context_output_layer = create_context_module(in_conv, 32, dropout_rate=dropout_rate)
    print(context_output_layer.shape)
    summation_layer = concatenate([in_conv, context_output_layer], axis=-1)
    print(summation_layer.shape)
    level_output_layers.append(summation_layer)
    print(len(level_output_layers))
    current_layer = summation_layer
    print(current_layer.shape)
    '''Layer2-End'''

    '''Layer3-Start'''
    in_conv = create_convolution_block(current_layer, 64, strides=(2, 2, 2))
    context_output_layer = create_context_module(in_conv, 64, dropout_rate=dropout_rate)
    print(context_output_layer.shape)
    summation_layer = concatenate([in_conv, context_output_layer], axis=-1)
    print(summation_layer.shape)
    level_output_layers.append(summation_layer)
    print(len(level_output_layers))
    current_layer = summation_layer
    print(current_layer.shape)
    '''Layer3-Stopped'''


    segmentation_layers = list()
    '''Segmentation Layers2'''
    up_sampling = create_up_sampling_module(current_layer, 64)
    print(up_sampling.shape)
    print("and")
    print(level_output_layers[1].shape)
    concatenation_layer = concatenate([level_output_layers[1], up_sampling], axis=-1)
    print(concatenation_layer.shape)
    localization_output = create_localization_module(concatenation_layer, 64)
    print(localization_output.shape)
    current_layer = localization_output

    '''Segmentation Layer2--END'''

    '''Segmentation Layer1--Start'''
    up_sampling = create_up_sampling_module(current_layer, 32)
    print(up_sampling.shape)
    print("and")
    print(level_output_layers[0].shape)
    concatenation_layer = concatenate([level_output_layers[0], up_sampling], axis=-1)
    print(concatenation_layer.shape)
    localization_output = create_localization_module(concatenation_layer, 32)
    print(localization_output.shape)
    current_layer = localization_output
    segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1),data_format="channels_last")(current_layer))
    print(segmentation_layers[0].shape)


    output_layer = None

    '''Layer1 Start'''
    segmentation_layer = segmentation_layers[0]
    output_layer = segmentation_layer
    activation_block = Activation(activation_name)(output_layer)
    print(output_layer.shape)

    model = Model(inputs=inputs, outputs=activation_block)

    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)

    return model
    '''
    parallel=multi_gpu_model(model,gpus=3)
    parallel.compile(optimizer(lr=initial_learning_rate),loss=loss_function)
    return parallel
    '''
    # for level_number in range(depth):
    #     n_level_filters = (2**level_number) * n_base_filters
    #     level_filters.append(n_level_filters)
    #
    #     if current_layer is inputs:
    #         in_conv = create_convolution_block(current_layer, n_level_filters)
    #     else:
    #         in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))
    #
    #     context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)
    #     print(context_output_layer.shape)
    #     #print(in_conv.size())
    #     summation_layer = concatenate([in_conv, context_output_layer],axis=-1)
    #     print(summation_layer.shape)
    #     level_output_layers.append(summation_layer)
    #     print(len(level_output_layers))
    #     current_layer = summation_layer
    #     print(current_layer.shape)
    # print("END")

    # for level_number in range(depth-2, -1, -1):
    #
    #     up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
    #
    #     concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)
    #     print(concatenation_layer.shape)
    #     localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
    #     current_layer = localization_output
    #     if level_number < n_segmentation_levels:
    #         segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    # for level_number in reversed(range(n_segmentation_levels)):
    #     segmentation_layer = segmentation_layers[level_number]
    #     if output_layer is None:
    #         output_layer = segmentation_layer
    #     else:
    #         output_layer = Add()([output_layer, segmentation_layer])
    #
    #     if level_number > 0:
    #         output_layer = Softmax()(output_layer) ###to softmax
    #
    # activation_block = Activation(activation_name)(output_layer)


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    #up_sample = Softmax(axis=-1)(input_layer)
    up_sample = UpSampling3D(size=size , data_format="channels_last")(input_layer)
    
    #print(len(input_layer))
    #print(up_sample.shape)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2

