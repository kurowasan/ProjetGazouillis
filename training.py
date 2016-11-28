__author__ = 'GazouillisTeam'

import numpy as np
import os
import time
from keras.callbacks import EarlyStopping
from preprocessing import batch_generator
from custom_logging import create_log, save_architecture, ModelSaver, trainargs2strings

def load_weights(model, path, h5py=False):
    if not h5py: # H5PY not available : save weights using np.save
        w = np.load(path)
        model.set_weights(w)
    else:
        model.load_weights(path)

def training(path, model, dataset, index_train, index_valid, D, batch_size,
             nsamples_per_epoch, nepoch, patience, lr, weighted_samples=False,
             pretrained=None, h5py=False):
    start = time.time()
    # Create dir (if not already done)
    if os.path.exists(path) is False:
        os.mkdir(os.path.abspath(path))
    path_weights = os.path.join(path, "weights")
    if os.path.exists(path_weights) is False:
        os.mkdir(os.path.abspath(path_weights))
    # Create log file
    if pretrained is None:
        settings = trainargs2strings(path, model, dataset, index_train, index_valid, D, batch_size,
                                     nsamples_per_epoch, nepoch, patience, lr)
        create_log(path, settings)
        # Save architecture
        save_architecture(model, path)
    else:
        load_weights(model, pretrained, h5py)
    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience)
    model_saver = ModelSaver(path, os.path.join(path, "weights"), monitor="val_loss", h5py=h5py)
    # Argument to give to generators
    train_generator_args = [dataset, index_train, batch_size, D, weighted_samples]
    valid_generator_args = [dataset, index_valid, 2*batch_size, D, weighted_samples]
    # Training loop
    h = model.fit_generator(batch_generator(*train_generator_args), nsamples_per_epoch, nepoch,
                            validation_data=batch_generator(*valid_generator_args),
                            nb_val_samples=len(index_valid),
                            callbacks=[early_stopping, model_saver])

    end = time.time()
    print "Training took %.2fs"%(end-start)