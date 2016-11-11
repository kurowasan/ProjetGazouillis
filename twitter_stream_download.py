# ###########################################################################################
# Code inspired from Marco Bonzanini's code:                                                #
# See https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/ and     #
# https://gist.github.com/bonzanini/af0463b927433c73784d                                    #
# ###########################################################################################

# To run this code, first edit config.py with your configuration, then:
#
# python twitter_stream_download.py -o output_directory -q queries -l language -t max_time -c max_count -k keyword -d description
# USUALLY : python twitter_stream_download.py -o tweets/ -l en -c 1000000 -k text
#
# It will produce the list of tweets for the query "apple"
# in the file data/stream_apple.json

import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
import twitter_config as config
import json


def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="Twitter Downloader")
    parser.add_argument("-q",
                        "--query",
                        dest="query",
                        help="Query/Filter",
                        default=list(set("abcdefghijklmnopqrstuvwxyz")))
    parser.add_argument("-o",
                        "--out-dir",
                        dest="out_dir",
                        help="Output/Data Directory")
    parser.add_argument("-l",
                        "--language",
                        dest="language",
                        help="Language/Filter",
                        default=None)
    parser.add_argument("-t",
                        "--timer",
                        dest="timer",
                        help="Timer/Limit in seconds",
                        default=3600)
    parser.add_argument("-c",
                        "--counter",
                        dest="counter",
                        help="Counter/Limit in # of tweets",
                        default=1e6)
    parser.add_argument("-d",
                        "--description",
                        dest="description",
                        help="Description/Header",
                        default="")
    parser.add_argument("-k",
                        "--keys",
                        dest="keys",
                        help="Keys/Filter",
                        default=None)
    return parser


def preprocess_tweet(tweet, keys=[]):
    if "<start>" in tweet or "<end>" in tweet:
        raise Exception("<start> or <end> already in the tweet")
    if len(keys) != 0:
        out = "<start>"
        json_tweet = json.loads(tweet)
        for k in keys:
            out += k + ":"
            out += json_tweet[k].encode('utf-8')
        out += "<end>\n"

        return out
    else:
        return tweet


class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self, data_dir, query=[], keys=[], description="", time_limit=3600, counter_limit=1e6):

        self.time_limit = time.time() + time_limit
        self.counter_limit = counter_limit
        self.counter = 0
        self.keys = keys
        self.outfile = "%s/stream_%s.txt" % (data_dir, time.strftime("%Y_%m_%d_%H-%M-%S"))
        self.create_out_file(query, keys, description)

    def create_out_file(self, query, keys, description):
        with open(self.outfile, 'w') as f:
            f.writelines(["Description: %s\n" % description,
                          "Queries: %s\n" % str(query),
                          "Keys: %s\n" % str(keys)])

    def on_data(self, data, text_only=False, verbose=True):
        if time.time() < self.time_limit and self.counter < self.counter_limit:
            self.counter += 1
            try:
                tweet = preprocess_tweet(data, self.keys)
                with open(self.outfile, 'a') as f:
                    f.writelines(tweet)
                    if verbose:
                        print(tweet)
                    return True
            except BaseException as e:
                print("Error on_data: %s" % str(e))
                time.sleep(0.1)
                print data
            return True
        else:
            return False

    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    """Convert file name into a safe string.

    Arguments:
        fname -- the file name to convert
    Return:
        String -- converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if invalid.

    Arguments:
        one_char -- the char to convert
    Return:
        Character -- converted char
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'


@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status


if __name__ == '__main__':
    # Get arguments
    parser = get_parser()
    args = parser.parse_args()

    # Queries
    if type(args.query) is list:
        queries = args.query
    else:
        if "_" in args.query:
            queries = args.query.split("_")
        else:
            queries = [args.query]

    # Keys
    if "_" in args.keys:
        keys = args.keys.split("_")
    else:
        keys = [args.keys]

    # Instantiate the streaming object
    auth = OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_token, config.access_secret)
    api = tweepy.API(auth)
    twitter_stream = Stream(auth, MyListener(args.out_dir,
                                             query=queries, keys=keys, description=args.description,
                                             time_limit=int(args.timer), counter_limit=int(args.counter)))

    # Stream Loop
    for i in range(1000):
        try:
            if args.language is None:
                # Track tweets
                twitter_stream.filter(track=queries)
            else:
                # Track tweets, with the appropriate language
                twitter_stream.filter(track=queries, languages=[args.language])
        except:
            # Wait a minute
            time.sleep(60)
