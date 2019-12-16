import keras
import keras_contrib
import tensorflow as tf


# Building blocks
def adding_conv(x, a, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, 
            activation=None, data_format=data_format)(x)
    c = keras.layers.add([c, a])
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c

def conv(x, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, 
            activation=None, data_format=data_format)(x)
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c

def down_conv(x, filters, kernel_size, padding, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=2, 
                            activation=None, data_format=data_format)(x)
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c

def up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    channel_axis = -1 if data_format=='channels_last' else 1
    c = keras.layers.Conv3DTranspose(filters, kernel_size=(2,2,2), strides=(2,2,2), 
                                    data_format=data_format)(x) # up dim(x) by x2
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(c)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, 
                            activation=None, data_format=data_format)(c)
    concat = keras.layers.Concatenate(axis=channel_axis)([c, skip]) # concat after Up; dim(skip) == 2*dim(x)
    c = keras_contrib.layers.GroupNormalization(groups=groups, axis=channel_axis)(concat)
    c = keras.layers.advanced_activations.PReLU()(c)
    return c


# Encoders
def encoder1(x, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('encoder1'):
        with tf.variable_scope('conv'):
            conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)
        with tf.variable_scope('addconv'):
            addconv = adding_conv(conv1, conv1, filters, kernel_size, padding, strides, data_format, groups) # N
        with tf.variable_scope('downconv'):
            downconv = down_conv(addconv, filters*2, kernel_size, padding, data_format, groups) # N/2
        return (addconv, downconv)

def encoder2(x, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('encoder2'):
        with tf.variable_scope('conv'):
            conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups) 
        with tf.variable_scope('addconv'):
            addconv = adding_conv(conv1, x, filters, kernel_size, padding, strides, data_format, groups) # N/2
        with tf.variable_scope('downconv'):
            downconv = down_conv(addconv, filters*2, kernel_size, padding, data_format, groups) # N/4
        return (addconv, downconv)

def encoder3(x, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('encoder3'):
        with tf.variable_scope('conv1'):
            conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups) # N/4
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups) # N/4
        with tf.variable_scope('addconv'):
            addconv = adding_conv(conv2, x, filters, kernel_size, padding, strides, data_format, groups) # N/4
        with tf.variable_scope('downconv'):
            downconv = down_conv(addconv, filters*2, kernel_size, padding, data_format, groups) # N/8
        return (addconv, downconv)

def encoder4(x, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('encoder4'):
        with tf.variable_scope('conv1'):
            conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups) # N/8
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups) # N/8
        with tf.variable_scope('addconv'):
            addconv = adding_conv(conv2, x, filters, kernel_size, padding, strides, data_format, groups) # N/8
        with tf.variable_scope('downconv'):
            downconv = down_conv(addconv, filters*2, kernel_size, padding, data_format, groups) # N/16
        return (addconv, downconv)


# Bottom
def bottom(x, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('bottom'):
        with tf.variable_scope('conv1'):
            conv1 = conv(x, filters, kernel_size, padding, strides, data_format, groups)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)
        with tf.variable_scope('addconv'):
            addconv = adding_conv(conv2, x, filters, kernel_size, padding, strides, data_format, groups) # N/16
        return addconv # N/16


# Decoders
def decoder4(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('decoder4'):
        with tf.variable_scope('upconv'):
            upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups) # N/8
        with tf.variable_scope('conv1'):
            conv1 = conv(upconv, filters, kernel_size, padding, strides, data_format, groups)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)
        return conv2 # N/8

def decoder3(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('decoder3'):
        with tf.variable_scope('upconv'):
            upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups) # N/4
        with tf.variable_scope('conv1'):
            conv1 = conv(upconv, filters, kernel_size, padding, strides, data_format, groups)
        with tf.variable_scope('conv2'):
            conv2 = conv(conv1, filters, kernel_size, padding, strides, data_format, groups)
        return conv2 # N/4

def decoder2(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('decoder2'):
        with tf.variable_scope('upconv'):
            upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups) # N/2
        with tf.variable_scope('conv'):
            conv1 = conv(upconv, filters, kernel_size, padding, strides, data_format, groups)
        return conv1 # N/2

def decoder1(x, skip, filters, kernel_size, padding, strides, data_format, groups):
    with tf.variable_scope('decoder1'):
        with tf.variable_scope('upconv'):
            upconv = up_conv_concat_conv(x, skip, filters, kernel_size, padding, strides, data_format, groups) # N
        return upconv # N


# Attention gate
def attention_gate(x, g, filters):
    with tf.variable_scope('attention_gate'):
        # Gating signal processing
        g = Conv3D(filters, kernel_size=1, data_format=data_format)(g)
        g = BatchNormalization()(g)

        # Skip signal processing: 
        x = Conv3D(filters, kernel_size=2, strides=2)(x) 
        x = BatchNormalization()(x1)

        # Add and proc
        g1_x1 = Add()([g1,x1])
        psi = Activation('relu')(g1_x1)
        psi = Conv3D(1, kernel_size = 1)(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        alpha = UpSampling3D(size=2)(psi)

        x_hat = Multiply()([x,alpha])
        return x_hat


# Model
def VNet(n_in, n_out, image_shape, filters, kernel_size, padding, strides, data_format, groups, inter_filters):
    with tf.variable_scope('VNet'):
        input_dim = image_shape+(n_in,) if data_format=='channels_last' \
            else (n_in,)+image_shape
        
        inputs = keras.layers.Input(input_dim)
        print('inputs', inputs.shape)

        (encoder1_addconv, encoder1_downconv) = encoder1(inputs, filters*2**0, kernel_size, padding, strides, data_format, groups) # N, N/2
        (encoder2_addconv, encoder2_downconv) = encoder2(encoder1_downconv, filters*2**1, kernel_size, padding, strides, data_format, groups) # N/2, N/4
        (encoder3_addconv, encoder3_downconv) = encoder3(encoder2_downconv, filters*2**2, kernel_size, padding, strides, data_format, groups) # N/4, N/8
        (encoder4_addconv, encoder4_downconv) = encoder4(encoder3_downconv, filters*2**3, kernel_size, padding, strides, data_format, groups) # N/8, N/16

        bottom_addconv = bottom(encoder4_downconv, filters*2**4, kernel_size, padding, strides, data_format, groups) # N/16

        encoder4_ag = attention_gate(encoder4_addconv, bottom_addconv, inter_filters) # (N/8, N/16) --> N/8
        decoder4_conv = decoder4(bottom_addconv, encoder4_ag, filters*2**3, kernel_size, padding, strides, data_format, groups) # N/8
        encoder3_ag = attention_gate(encoder3_addconv, decoder4_conv, inter_filters) # (N/4, N/8) --> N/4
        decoder3_conv = decoder3(decoder4_conv, encoder3_ag, filters*2**2, kernel_size, padding, strides, data_format, groups) # N/4
        encoder2_ag = attention_gate(encoder2_addconv, decoder3_conv, inter_filters) # (N/2, N/4) --> N/2
        decoder2_conv = decoder2(decoder3_conv, encoder2_ag, filters*2**1, kernel_size, padding, strides, data_format, groups) # N/2
        encoder1_ag = attention_gate(encoder1_addconv, decoder2_conv, inter_filters) # (N, N/2) --> N
        decoder1_conv = decoder1(decoder2_conv, encoder1_ag, filters*2**0, kernel_size, padding, strides, data_format, groups) # N
       
        with tf.variable_scope("output"):
            outputs = keras.layers.Conv3D(n_out,
                (1,1,1),
                padding='same',
                activation='sigmoid',
                data_format=data_format)(decoder1_conv)

            model = keras.models.Model(inputs, outputs)

        return model

