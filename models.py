__author__ = 'GazouillisTeam'

from keras.models import Model
from keras.layers import Input, LSTM, Masking, Dropout, TimeDistributed, Dense, Activation, ZeroPadding1D, \
    BatchNormalization, AtrousConvolution1D
from keras.optimizers import Adam

def get_LSTM_v1(T, D, lr, nhidden, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # Masking "only-0" input features
    masked = Masking(mask_value=0.0)(inputs)
    # Hidden layers
    for i in range(nhidden):
        if i == 0:
            hidden  = LSTM(128, return_sequences=True)(masked)
        else:
            hidden  = LSTM(128, return_sequences=True)(dropout)
        dropout = Dropout(drop_rate)(hidden)
    # Output layer : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(dropout) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model

def CausalConvolution1D(input_layer, nfilters, filter_length, atrous_rate=1, activation="linear", **kwargs):
    total_length = filter_length + (filter_length-1)*(atrous_rate-1)
    # Asymetric padding : 0 added only on the left side
    padd = ZeroPadding1D((total_length-1,0))(input_layer)
    # Convolution
    conv = AtrousConvolution1D(nfilters, filter_length, atrous_rate=atrous_rate, border_mode='valid', **kwargs)(padd)
    bn = BatchNormalization()(conv)
    activ = Activation(activation)(bn)
    # Return
    return activ

def get_CausalCNN_v1(T, D, lr, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # Masking "only-0" input features : ZeroPadding do not support Masking
    # masked = Masking(mask_value=0.0)(inputs)
    # Hidden layers
    for i in range(10):
        if i == 0:
            hidden  = CausalConvolution1D(inputs, 128, 11, atrous_rate=1, activation="relu")
        else:
            hidden  = CausalConvolution1D(dropout, 128, 11, atrous_rate=1, activation="relu")
        dropout = Dropout(drop_rate)(hidden)
    # Output layer : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(dropout) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model