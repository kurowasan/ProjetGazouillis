__author__ = 'GazouillisTeam'

from keras.models import Model
from keras.layers import Input, LSTM, Masking, Dropout, TimeDistributed, Dense, Activation, ZeroPadding1D, \
    BatchNormalization, AtrousConvolution1D, merge, Convolution1D
from keras.optimizers import Adam

def get_LSTM_v1(T, D, lr, nhidden, nneuronsh, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # Masking "only-0" input features
    masked = Masking(mask_value=0.0)(inputs)
    # Hidden layers
    for i in range(nhidden):
        if i == 0:
            hidden  = LSTM(nneuronsh, return_sequences=True)(masked)
        else:
            hidden  = LSTM(nneuronsh, return_sequences=True)(dropout)
        dropout = Dropout(drop_rate)(hidden)
    # Output layer : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(dropout) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model

def CausalConvolution1D(input_layer, nfilters, filter_length, atrous_rate=1, activation="linear", batch_norm=True,
                        **kwargs):
    total_length = filter_length + (filter_length-1)*(atrous_rate-1)
    # Asymetric padding : 0 added only on the left side
    padd = ZeroPadding1D((total_length-1,0))(input_layer)
    # Convolution
    conv = AtrousConvolution1D(nfilters, filter_length, atrous_rate=atrous_rate, border_mode='valid', **kwargs)(padd)
    if batch_norm:
        bn = BatchNormalization()(conv)
        activ = Activation(activation)(bn)
    else:
        activ = Activation(activation)(conv)
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

def get_CausalCNN_v2(T, D, lr, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # Masking "only-0" input features : ZeroPadding do not support Masking
    # masked = Masking(mask_value=0.0)(inputs)
    # Hidden layers
    for i in range(3):
        if i == 0:
            hidden  = CausalConvolution1D(inputs, 128, 3, atrous_rate=1, activation="linear")
            hidden  = CausalConvolution1D(hidden, 128, 5, atrous_rate=2, activation="relu")
        else:
            hidden  = CausalConvolution1D(hidden, 128, 3, atrous_rate=1, activation="linear")
            hidden  = CausalConvolution1D(hidden, 128, 5, atrous_rate=2, activation="relu")
        dropout = Dropout(drop_rate)(hidden)
    # Output layer : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(dropout) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model

def get_CausalCNN_v3(T, D, lr, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # Masking "only-0" input features : ZeroPadding do not support Masking
    # masked = Masking(mask_value=0.0)(inputs)
    # Hidden layers
    for i in range(3):
        if i == 0:
            lin_hidden  = CausalConvolution1D(inputs, 128, 3, atrous_rate=1, activation="linear")
        else:
            lin_hidden  = CausalConvolution1D(hidden, 128, 3, atrous_rate=1, activation="linear")
        #
        tanh_hidden  = CausalConvolution1D(lin_hidden, 128, 5, atrous_rate=2, activation="tanh")
        sigm_hidden  = CausalConvolution1D(tanh_hidden, 128, 5, atrous_rate=2, activation="sigmoid")
        hidden = merge([tanh_hidden, sigm_hidden], mode="mul")#, output_shape=(160,128))
        #
        dropout = Dropout(drop_rate)(hidden)
    # Output layer : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(dropout) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model

def get_CausalCNN_v4(T, D, lr, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # First block : CausalCNN layers
    hidden  = CausalConvolution1D(inputs, 128, 3, atrous_rate=1, activation="tanh")
    for i in range(3):
        hidden  = CausalConvolution1D(hidden, 64, 3, atrous_rate=1, activation="linear", batch_norm=False)
        hidden  = CausalConvolution1D(hidden, 128, 3, atrous_rate=2**(i+1), activation="tanh")
    # Second block : Residual layers
    for i in range(5):
        # Convolution 1x1 to reduce complexity
        projection = Convolution1D(64, 1, activation='linear')(hidden)
        # Causal CNN layers :
        tanh_hidden  = CausalConvolution1D(projection, 128, 5, atrous_rate=2**(i+1), activation="tanh")
        sigm_hidden  = CausalConvolution1D(projection, 128, 5, atrous_rate=2**(i+1), activation="sigmoid")
        # Merge : tanh * sigmoid (see WaveNet paper)
        activ = merge([tanh_hidden, sigm_hidden], mode="mul")#, output_shape=(160,128))
        # Residual merge
        hidden = merge([hidden, activ], mode="sum")
    # Last block : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(hidden) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model

def get_CausalCNN_v5(T, D, lr, drop_rate):
    # Input layer
    inputs = Input((T, D))
    # First block : CausalCNN layers
    hidden  = CausalConvolution1D(inputs, 128, 3, atrous_rate=1, activation="tanh")
    for i in range(3):
        hidden  = CausalConvolution1D(hidden, 256, 3, atrous_rate=1, activation="linear", batch_norm=False)
        hidden  = CausalConvolution1D(hidden, 128, 3, atrous_rate=2**(i+1), activation="tanh")
    # Second block : Residual layers
    for i in range(5):
        # Causal CNN layers :
        tanh_hidden  = CausalConvolution1D(hidden, 256, 5, atrous_rate=2**(i+1), activation="tanh")
        sigm_hidden  = CausalConvolution1D(hidden, 256, 5, atrous_rate=2**(i+1), activation="sigmoid")
        # Merge : tanh * sigmoid (see WaveNet paper)
        activ = merge([tanh_hidden, sigm_hidden], mode="mul")#, output_shape=(160,128))
        # Convolution 1x1 to reduce complexity
        projection = Convolution1D(128, 1, activation='linear')(activ)
        # Residual merge
        hidden = merge([hidden, projection], mode="sum")
    # Last block : linear TimeDistributedDense + softmax
    decoder = TimeDistributed(Dense(D))(hidden) # Apply the same dense layer on each timestep
    outputs = Activation("softmax") (decoder)

    model = Model(input=inputs, output=outputs)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy")

    return model