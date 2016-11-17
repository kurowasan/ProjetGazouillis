__author__ = 'GazouillisTeam'

import os
import time
from keras.callbacks import EarlyStopping
from preprocessing import batch_generator
from custom_logging import create_log, save_architecture, ModelSaver, trainargs2strings

def training(path, model, dataset, index_train, index_valid, D, batch_size,
             nsamples_per_epoch, nepoch, patience): # see training.py
    start = time.time()
    # Create dir (if not already done)
    if os.path.exists(path) is False:
        os.mkdir(os.path.abspath(path))
    path_weights = os.path.join(path, "weights")
    if os.path.exists(path_weights) is False:
        os.mkdir(os.path.abspath(path_weights))
    # Create log file
    settings = trainargs2strings(path, model, dataset, index_train, index_valid, D, batch_size,
                                 nsamples_per_epoch, nepoch, patience)
    create_log(path, settings)
    # Save architecture
    save_architecture(model, path)
    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience)
    model_saver = ModelSaver(path, os.path.join(path, "weights"), monitor="val_loss", h5py=False)
    # Argument to give to generators
    train_generator_args = [dataset, index_train, batch_size, D]
    valid_generator_args = [dataset, index_valid, 2*batch_size, D]
    # Training loop
    h = model.fit_generator(batch_generator(*train_generator_args), nsamples_per_epoch, nepoch,
                            validation_data=batch_generator(*valid_generator_args),
                            nb_val_samples=len(index_valid),
                            callbacks=[early_stopping, model_saver])

    end = time.time()
    print "Training took %.2fs"%(end-start)