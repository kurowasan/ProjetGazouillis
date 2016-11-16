__author__ = 'GazouillisTeam'

### Encoding params ###
ACCEPTED_CHARS = ['\n', ' ', '!', '"', '#', '&', "'", '(', ')', '*', ',', '-', '.', '/',
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', '@', '_',
                  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                  'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                  '\x80', '\x92', '\x98', '\x9f', '\xa6', '\xe2', '\xf0']
T = 160
D = 64 # 62 chars + 1 extra char to indicate to 'end' of a tweet + 1 extra dim for padding

### Training params ###
LR = 1e-3 # learning rate
B  = 256  # batch size
NHIDDEN = 3
DROPOUT_RATE = 0.1
PATH_DATA = "/data/lisa/exp/bergegu/gazouillis/data/dataset.npy"
PATH_EXPERIMENT = "/data/lisa/exp/bergegu/gazouillis/experiments/lstm_h3_n128"
NB_EPOCHS = 1000
NB_SAMPLES_PER_EPOCH = B*int(5e5 / B) # around 500 000
PATIENCE = 20