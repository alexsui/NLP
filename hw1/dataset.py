import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import nltk
import random
import re                                 
import string
from cleantext import clean            
from nltk.corpus import stopwords     
from nltk.stem import PorterStemmer       
from nltk.tokenize import TweetTokenizer 
from textaugment import Translate, EDA
class myDataset(Dataset):
    def __init__(self, root_dir, vocab, max_seq_length, pad_token, unk_token, kind ="elmo",mode = "train", transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.stopwords = stopwords.words('english')
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.kind = kind
        self.max_seq_length = max_seq_length
        self.w2i = {term:i for i, term in enumerate(vocab)}
        self.i2w = {i:term for i, term in enumerate(vocab)}
        self.data = self.read_data(root_dir)              
        self.tweets = self.preprocess(self.data.tweet)
        if self.mode != "test":
            self.labels = self.data.labels.tolist()
        self.classes = list(pd.read_csv("./HW1_dataset/sample_submission.csv").columns[1:])
        if self.kind =="elmo":
            self.train_embedding = np.load("./elmo_X_train.npy", allow_pickle=True)
            self.train_embedding = np.concatenate(self.train_embedding, axis=1)
            self.train_embedding = np.transpose(self.train_embedding, (1,0,2))
            self.test_embedding = np.load("./elmo_X_test.npy", allow_pickle=True)
            self.test_embedding = np.concatenate(self.test_embedding, axis=1)
            self.test_embedding = np.transpose(self.test_embedding, (1,0,2))
            self.val_embedding = np.load("./elmo_X_val.npy", allow_pickle=True)
            self.val_embedding = np.concatenate(self.val_embedding, axis=1)
            self.val_embedding = np.transpose(self.val_embedding, (1,0,2))
            self.train_label = np.load("./elmo_y_train.npy", allow_pickle=True)
            self.train_label = np.vstack(self.train_label)
            self.val_label = np.load("./elmo_y_val.npy", allow_pickle=True)
            self.val_label = np.vstack(self.val_label)
        elif self.kind == "elmo_new":
            self.train_embedding = np.load("./elmo_embedding/new_preprocess_elmo_X_train.npy", allow_pickle=True)
            self.val_embedding = np.load("./elmo_embedding/new_preprocess_elmo_X_val.npy", allow_pickle=True)
            self.test_embedding = np.load("./elmo_embedding/new_preprocess_elmo_X_test.npy", allow_pickle=True)
            self.train_label = np.load("./elmo_embedding/new_preprocess_elmo_y_train.npy", allow_pickle=True)
            self.val_label = np.load("./elmo_embedding/new_preprocess_elmo_y_val.npy", allow_pickle=True)
        self.tokenizer = tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    def convert_text_to_input_ids(self, text,pad_to_len):
        words = text[:pad_to_len]
        pad_len = pad_to_len - len(words)
        words.extend([self.pad_token]*pad_len)
        for i in range(len(words)):
            if words[i] not in self.w2i:
                words[i] = self.w2i[self.unk_token]
            else:
                words[i] = self.w2i[words[i]]
        return torch.Tensor(words).long()
    def preprocess(self, tweets):
        def filter_words(x):
            x = re.sub(r'@[A-Za-z0-9_]+', '', x)
            x = re.sub(r'[@#][^\s]*\s*|\s*[@#][^\s]*$', '', x)
            x = re.sub(r'RT[\s]+', '', x)
            x = re.sub(r"@([A-Za-z0-9_]{4,15})", r"@ <user>", x)
            x = re.sub(r'\$', '', x)
            x = re.sub(r'\+', '', x)
            x = re.sub(r'\|', '', x)
            x = re.sub(r'\.\.\.', '', x)
            # x = x.replace("#","<hashtag>")
            x = clean(x,no_emoji=True, lower=True, no_urls =True, no_numbers=True, replace_with_url="", replace_with_number="<NUMBER>")
            return x
        
        # TODO : stemming
        filtered_tweets = tweets.apply(lambda x : filter_words(x))
        return filtered_tweets
    def encode_label(self, label):
        target = torch.zeros(12)
        for l in label:
            idx = self.classes.index(l)
            target[idx] = 1
        return target
    def read_data(self, root_dir):
        with open(root_dir, "r") as f:
            data = json.load(f)
        df =  pd.DataFrame(data)
        if self.mode != "test":
            df["labels"] = df.labels.apply(lambda x : list(x.keys()))
        return df 
    def augment(self, tweet, prob=0.5):
        t = EDA()
        t2 = Translate(src="en", to="nl")
        if random.random() > prob:
            return tweet
        if random.random() > 0.5:
            return t.synonym_replacement(tweet)
        else:
            return t2.augment(tweet)
    def __len__(self):
        if self.kind == "elmo" or self.kind == "elmo_new":
            if self.mode =="train":
                return len(self.train_embedding)
            elif self.mode == "valid":
                return len(self.val_embedding)
            elif self.mode == "test":
                return len(self.test_embedding)
        else:
            return len(self.tweets)
    def __getitem__(self, idx):
        if self.kind == "elmo" or self.kind == "elmo_new":
            if self.mode =="train":
                return torch.tensor(self.train_embedding[idx],dtype=torch.float32), torch.tensor(self.train_label[idx])
            elif self.mode == "valid":
                return  torch.tensor(self.val_embedding[idx],dtype=torch.float32),  torch.tensor(self.val_label[idx])
            elif self.mode == "test":
                return  torch.tensor(self.test_embedding[idx],dtype=torch.float32)
        else:
            tweet = self.tweets[idx]
            if self.transform:
                tweet = self.augment(tweet)
            tweet_token = tweet.apply(lambda x : self.tokenizer.tokenize(x))
            tweet_token = tweet_token.apply(lambda x : [word for word in x if word not in self.stopwords])
            tweet_token = tweet_token.apply(lambda x :self.convert_text_to_input_ids(x, self.max_seq_length) )
            if self.mode == "test":
                return tweet
            else:
                label = self.labels[idx]
                return tweet, self.encode_label(label)