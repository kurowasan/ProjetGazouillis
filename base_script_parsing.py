__author__ = 'GazouillisTeam'

import numpy as np
import params
import models
import preprocessing
import training as tr

# Load dataset
dataset = np.load(params.PATH_DATA)

# Splitting the dataset
index_train, index_valid, index_test = preprocessing.split_dataset(dataset)

# Define the model (frozen + pretrained)
model = models.get_frozen_LSTM(params.T-1, params.D-1, 51, params.LR, params.NHIDDEN, params.NNEURONSH, params.DROPOUT_RATE,
                               params.PRETRAINED, params.H5PY)
#model = models.get_CausalCNN_v5(params.T-1, params.D-1, params.LR, params.DROPOUT_RATE)

# Training
tr.training_on_parsed_data(params.PATH_EXPERIMENT, model, dataset, index_train, index_valid,
                           params.D, params.T, params.B, params.NB_SAMPLES_PER_EPOCH, params.NB_EPOCHS, params.PATIENCE,
                           params.LR, params.ACCEPTED_CHARS, params.ACCEPTED_TAGS_DICT,
                           weighted_samples=params.WEIGHTED_SAMPLES, pretrained=None, h5py=params.H5PY)