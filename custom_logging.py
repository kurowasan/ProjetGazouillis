__author__ = 'GazouillisTeam'

import numpy as np
import os
import sys
import time

from keras.callbacks import Callback

def save_architecture(model, path_out):
    """
    Based on the keras utils 'model.summary()'
    """
    # Redirect the print output the a textfile
    orig_stdout = sys.stdout
    # and store the architecture
    f = file(os.path.join(path_out, "architecture.txt"), 'w')
    sys.stdout = f
    model.summary()
    # Reset the print output direction
    sys.stdout = orig_stdout
    f.close()

    open(os.path.join(path_out, "config.json"), 'w').write(model.to_json())

def create_log(path, settings, filename="log.txt"):
    f = open(os.path.join(path, filename), "w")
    f.writelines(str(settings))
    f.writelines("\n####\nStarted on %s at %s\n" % (time.strftime("%d/%m/%Y"), time.strftime("%H:%M:%S")))
    f.close()

def write_log(path, string, filename="log.txt"):
    """
    Add a line at the end of a textfile.

    :param path: textfile location
    :param string: line to add
    """
    # Open and Read
    f = open(os.path.join(path, filename), "r")
    lines = f.readlines()
    f.close()
    # Adding a line
    lines.append(string)
    # Write
    f = open(os.path.join(path, filename), "w")
    f.writelines(lines)
    f.close()

class ModelSaver(Callback):
    """
    Keras callback subclass which defines a saving procedure of the model being trained : after each epoch,
    the last model is saved under the name 'after_random.cnn'. The best model is saved with the name 'best_model.cnn'.
    The model after random can also be saved. And the model architecture is saved with the name 'config.network'.
    Everything is stored using pickle.
    """

    def __init__(self, path, path_weights, monitor, verbose=1):
        super(Callback, self).__init__()
        self.verbose = verbose
        self.path = path
        self.path_weights = path_weights
        self.monitor = monitor
        self.best = np.Inf


    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_end = time.time()
        # get loss
        monitor = logs.get(self.monitor)
        # condition = True if loss decreased
        condition = monitor < self.best

        if condition:
            # Save weights as "best_model.weights"
            self.best = monitor
            save_path = os.path.join(self.path_weights, "best_model.weights")
            self.model.save_weights(save_path, overwrite=True)
        else:
            # Save weights as "last_epoch.weights"
            save_path = os.path.join(self.path_weights, "last_epoch.weights")
            self.model.save_weights(save_path, overwrite=True)

        # Log file management
        if self.verbose > 0:
            log_string = "####\nEpoch %d took %d s: " % (epoch, int(self.epoch_end-self.epoch_start))
            for k in logs.keys():
                log_string += "%s : %.4f # " % (k, logs.get(k))
            if condition:
                log_string += "\tBEST"
            write_log(self.path, log_string)

def trainargs2strings(path, model, dataset, index_train, index_valid, D, batch_size,
             nsamples_per_epoch, nepoch, patience):
    settings = ""
    settings += "Path : %s"%path
    settings += "\nDataset shape :" + str(dataset.shape)
    settings += "\nNtrain : %d"%len(index_train)
    settings += "\nNvalid : %d"%len(index_valid)
    settings += "\nDim : %d"%D
    settings += "\nBatch size : %d"%batch_size
    settings += "\nNb samples per epoch : %d"%nsamples_per_epoch
    settings += "\nNb epochs : %d"%nepoch
    settings += "\nPatience : %d"%patience
    return settings