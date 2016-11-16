__author__ = 'GazouillisTeam'

from keras.models import Model
from keras.layers import Input, LSTM, Masking, Dropout, TimeDistributed, Dense, Activation
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