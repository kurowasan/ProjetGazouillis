__author__ = 'GazouillisTeam'

import numpy as np
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import tokenize
from nltk import TweetTokenizer

def split_dataset(dataset, split_rate=[0.95, 0.01, 0.04], seed=123):
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
    one_hot_batch = np.zeros((B*T, D), "float32")
    one_hot_batch[range(B*T), batch.flatten()] = 1
    one_hot_batch = one_hot_batch.reshape((B,T,D))
    return one_hot_batch

def batch2tweet(batch, accepted_caracters, special_char=""): # see preprocessing.py
    '''Not optimized. But not used during the training : no need to be fast.'''
    tweets = []
    for t in batch:
        tweet = ""
        for char in t:
            try:
                tweet += accepted_caracters[char[0]]
            except:
                tweet += special_char # Special marker indicating the end of the tweet
        tweets.append(tweet.decode("unicode-escape"))
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

def token_position(tweet, tags):
    '''Get the position of next word using nltk tokenizing functions'''
    tknzr = TweetTokenizer()

    output = []
    offset, length, offset2, length = 0, 0, 0, 0
    for i,sentence in enumerate(tknzr.tokenize(tweet)):
        # fix ignored characters
        offset = tweet.find(sentence, offset)
        length = len(sentence)
        output.append([offset, length, tags[i]])
        offset += length
    return output

def string2label(tweet, accepted_tags_dict, T):
    '''String parsing : tweet to character classification'''
    tknzr = TweetTokenizer()
    token = tknzr.tokenize(tweet) # liste des mots du tweet
    tag = nltk.pos_tag(token) # [('mot','classe'),...]
    D = len(accepted_tags_dict.keys()) # construction de la sortie de maniere sequentielle (mot par mot)
    output = []
    start_pos = 0
    start_pos_next = 0
    for i,t in enumerate(tag):
        word, tclasse = t
        tclasse = accepted_tags_dict[tclasse]
        if word.find("rt")>=0:
            tclasse = accepted_tags_dict["RT"]
        if word.find("#")>=0:
            tclasse = accepted_tags_dict["#"]
        if word.find("@")>=0:
            tclasse = accepted_tags_dict["@"]
        if word.find("http")>=0:
            tclasse = accepted_tags_dict["HTTP"]
        output.extend([tclasse]*len(word)) # ajoute le bon nombre de fois la classe
        # Espace
        start_pos = tweet.find(word, start_pos_next)
        try:
            start_pos_next = tweet.find(tag[i+1][0], start_pos + len(word))
            # Nb d'espace
            nb_space = start_pos_next - start_pos - len(word)
            # Ajouter les espaces
            output.extend([D]*nb_space)
        except:
            pass
    # Padding
    if len(output) < T:
        output.extend([D+1]*(T-len(output))) #### BUG ####
    return output

def batch2label(batch, T, accepted_caracters, accepted_tags_dict):
    '''String parsing : tweet to character classification'''
    outputs = []
    tweets = batch2tweet(batch, accepted_caracters, special_char="")
    # String2label conversion (tweet by tweet)
    for tweet in tweets:
        conversion = string2label(tweet, accepted_tags_dict, T)
        if len(conversion) != T:
            conversion = [int(len(accepted_tags_dict.keys())+1) for i in range(T)]
        outputs.append(conversion)
    outputs = np.array(outputs)
    D = len(accepted_tags_dict.keys())
    batch = batch2onehot(outputs, D+2)
    return batch

def batch_generator(data, index, batch_size, D, weighted_samples=False):
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
        if not weighted_samples:
            yield input_batch, target_batch
        else:
            # Weights = T / nb_of_non-zero_chararecter
            sample_weights = float(input_batch.shape[1]) / input_batch.sum(axis=2).sum(axis=1)
            yield input_batch, target_batch, sample_weights


def parsed_batch_generator(data, index, batch_size, D, T, accepted_caracters, accepted_tags_dict,
                           weighted_samples=False):
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
        # Input
        input_batch  = one_hot_batch[:, 0:-1, :]
        # Target
        one_hot_target = batch2label(raw_batch, T, accepted_caracters, accepted_tags_dict)
        target_batch = one_hot_target[:, 1:, :] # target = 1-shifted input batch
        del raw_batch
        count += batch_size
        # Yield
        if not weighted_samples:
            yield input_batch, target_batch
        else:
            # Weights = T / nb_of_non-zero_chararecter
            sample_weights = float(input_batch.shape[1]) / input_batch.sum(axis=2).sum(axis=1)
            yield input_batch, target_batch, sample_weights


def multi_batch_generator(data, index, batch_size, D, T, accepted_caracters, accepted_tags_dict,
                           weighted_samples=False):
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
        one_hot_tags = batch2label(raw_batch, T, accepted_caracters, accepted_tags_dict)
        # Input
        text_input_batch  = one_hot_batch[:, 0:-1, :]
        tags_input_batch  = one_hot_tags[:, 0:-1, :]
        # Target
        text_target_batch = one_hot_batch[:, 1:, :] # target = 1-shifted input batch
        tags_target_batch = one_hot_tags[:, 1:, :]
        del raw_batch
        count += batch_size
        # Yield
        if not weighted_samples:
            yield [text_input_batch, tags_input_batch], [text_target_batch, tags_target_batch]
        else:
            # Weights = T / nb_of_non-zero_chararecter
            sample_weights = float(text_input_batch.shape[1]) / text_input_batch.sum(axis=2).sum(axis=1)
            yield [text_input_batch, tags_input_batch], [text_target_batch, tags_target_batch]