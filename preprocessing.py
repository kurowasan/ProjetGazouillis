__author__ = 'GazouillisTeam'


import numpy as np

def split_dataset(dataset, split_rate=[0.9, 0.05, 0.05], seed=123):
    # index = 1,2,3,...,N
    N = dataset.shape[0]
    index = range(dataset.shape[0])
    # shuffle
    np.random.seed(seed)
    np.random.shuffle(index)
    # division
    train_rate, valid_rate, test_rate = np.cumsum(split_rate)
    index_train = index[0:int(train_rate*N)]
    index_valid = index[int(train_rate*N):int(valid_rate*N)]
    index_test = index[int(valid_rate*N):int(test_rate*N)]
    # return
    return index_train, index_valid, index_test

def batch2onehot(batch, D):
    ''' Function used during the training to encode batches.
    Input size : (batch_size, tweet_length, 1).
    Output size : (batch_size, tweet_length, D)'''
    B, T = batch.shape[0:2]
    one_hot_batch = np.zeros((B*T, D))
    one_hot_batch[range(B*T), batch.flatten()] = 1
    one_hot_batch = one_hot_batch.reshape((B,T,D))
    return one_hot_batch

def batch2tweet(batch, accepted_caracters, special_char=""):
    '''Not optimized. But not used during the training : no need to be fast.'''
    tweets = []
    for t in batch:
        tweet = ""
        for char in t:
            try:
                tweet += accepted_caracters[char[0]]
            except:
                tweet += special_char # Special marker indicating the end of the tweet
        tweets.append(tweet)
    return tweets

def onehot2tweet(batch, accepted_caracters, special_char=""):
    '''Not optimized. But not used during the training : no need to be fast.'''
    tweets = []
    for t in batch:
        tweet = ""
        for char in t:
            try:
                tweet += accepted_caracters[np.where(char==1)[0][0]]
            except:
                tweet += special_char # Special marker indicating the end of the tweet
        tweets.append(tweet)
    return tweets

def batch_generator(data, index, batch_size, D):
    # Init iterator and shuffling the dataset
    count = 0
    np.random.shuffle(index)
    while 1:
        if count+batch_size >= len(index):
            # Reset counter and shuffling
            count = 0
            np.random.shuffle(index)
        # Get raw data
        raw_batch = np.copy(data[index[count:(count+batch_size)]])
        # One-hot encoding
        one_hot_batch = batch2onehot(raw_batch, D)
        # Remove the padding dimension
        one_hot_batch = one_hot_batch[:,:,0:(D-1)] # s.t. padding features are full of 0s
                                                   # and will be masked by the Masking layer
                                                   # (see below in the model definition)
        # Target
        input_batch  = one_hot_batch[:, 0:-1, :]
        target_batch = one_hot_batch[:, 1:, :] # target = 1-shifted input batch
        del raw_batch
        count += batch_size
        # Yield
        yield input_batch, target_batch