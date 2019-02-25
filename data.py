import os
import csv
import sys
import re
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch import functional as F

MOVIE_CONVERSATIONS = 'movie_conversations.txt'
MOVIE_LINES = 'movie_lines.txt'

def get_sentences(data_dir, max_len, min_count):
    """Return lists of sentences of list of words for the inputs and targets"""
    inputs, targets = get_raw_data(data_dir)
    inputs, targets = process_raw_data(inputs, targets, max_len, min_count)
    inputs, targets = sort_raw_data(inputs, targets)
    return inputs, targets

def sort_raw_data(inputs, targets):
    sdata = []
    for i, t in zip(inputs, targets):
        sdata.append([i,t])
    sdata.sort(key=lambda x: len(x[0]), reverse=True)
    return zip(*sdata)

def process_raw_data(raw_inputs, raw_targets, max_len, min_count=5000):
    """Return proccessed data:
        1) Strip multiple whitespaces
        2) Remove any special characters not in ['!', '.', '?', ";", ',']
        3) Remove sentences with rare words
        4) Remove input, target pairs with more than MAX_LEN words per
           sentence
        5) Lower case
    """
    #1,#2
    sinputs = []
    stargets = []
    for i, d in enumerate(zip(raw_inputs, raw_targets)):
        sinputs.append([ri for ri in d[0] if ri!=''])
        stargets.append([rt for rt in d[1] if rt!=''])
        temp = []
        for j in range(len(d[0])):
            d[0][j] = d[0][j].lower()
            temp.extend(re.findall(r"[\w']+|[.,!?;]", d[0][j]))
        sinputs[i] = temp.copy()
        temp = []
        for j in range(len(d[1])):
            d[1][j] = d[1][j].lower()
            temp.extend(re.findall(r"[\w']+|[.,!?;]", d[1][j]))
        stargets[i] = temp.copy()
    #3
    allWords = [w for s in sinputs+stargets for w in s]
    count = Counter(allWords)
    ts = []
    tt = []
    for si, st in zip(sinputs, stargets):
        common = True
        for w in si:
            if count[w]<min_count:
                common = False
                break
        if common:
            ts.append(si)
            tt.append(st)
    sinputs = ts.copy()
    stargets = tt.copy()
    ts = []
    tt = []
    for si, st in zip(sinputs, stargets):
        common = True
        for w in st:
            if count[w]<min_count:
                common = False
                break
        if common:
            ts.append(si)
            tt.append(st)
    sinputs = ts.copy()
    stargets = tt.copy()
    #4 
    sinputs, stargets = get_trimmed_data(sinputs, stargets, max_len)
    return sinputs, stargets


def get_raw_data(data_dir):
    """Return RAW_INPUTS and RAW_TARGETS as lists of sentences (str format)."""
    data = get_lines(data_dir+MOVIE_LINES)
    convs = get_conversations(data, data_dir+MOVIE_CONVERSATIONS)
    convpairs = get_convpairs(convs)
    raw_inputs, raw_targets = zip(*convpairs)
    return raw_inputs, raw_targets

def get_lines(movie_lines):
    with open(movie_lines, 'r', encoding='iso-8859-1') as f:
        text = f.readlines()
    lines = {}
    for line in text:
        val = line.split(" +++$+++ ")
        lines[val[0]] = {'lineID': val[0], 
                         'charID': val[1], 
                         'movieID': val[2], 
                         'char': val[3], 
                         'text': val[4]}
    return lines

def get_conversations(data, movie_conversations):
    with open(movie_conversations, 'r', encoding='iso-8859-1') as f:
        movie_convs = f.readlines()
    convs = []
    for movie in movie_convs:
        lineIDS = eval(movie.split(" +++$+++ ")[-1])
        text = []
        for lineID in lineIDS:
            text.append(data[lineID]['text'])
        convs.append(text)
    return convs

def get_convpairs(convs):
    convpairs = []
    for conv in convs:
        for i in range(len(conv)-1):
            query = conv[i].strip().split(" ")
            response = conv[i+1].strip().split(" ")
            convpairs.append([query, response])
    return convpairs

def get_trimmed_data(sinputs, stargets, max_len):
    sit = []
    stt = []
    for si, st in zip(sinputs, stargets):
        if len(si)<=max_len and len(st)<=max_len-1 and len(si)>0 and len(st)>0:
            sit.append(si)
            st.append("<END>")
            stt.append(st)
    return sit, stt

def get_tensors(vocab, sdata, max_len=20):
    """Return tensors with shape (nwords, nsentences)"""
    data = []
    masks = []
    pad_id = vocab.get_special_token('PAD')
    for sd in sdata:
        # initialise array with special pad_id
        arr = torch.full([max_len, len(sd)], pad_id, dtype=torch.int64)
        l = []
        for i, sentence in enumerate(sd):
            col = [vocab[idx_word] for idx_word in sentence]
            arr[:len(col), i] = torch.tensor(col, dtype=torch.int64)
            l.append(len(col))
        data.append(arr)
        data.append(torch.tensor(l, dtype=torch.int64))
    return data


def get_tokens_from_data(sinputs, stargets):
    return list(set([w for s in list(sinputs)+list(stargets) for w in s]))

def get_tokens_from_vectors_file(vectors_file):
    tokens = []
    vectors = []
    with open(vectors_file, 'r') as f:
        for l in f:
            line = l.split()
            tokens.append(line[0])
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    lv = len(vectors)
    arr = np.zeros([lv, vectors[0].shape[0]])
    for i in range(lv):
        arr[i, :] = vectors[i]
    return tokens, torch.from_numpy(arr)

def get_tokens_from_checkpoint_file(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    return checkpoint['tokens']

def clear_not_found(tokens, sinputs, stargets):
    inputs = []
    targets = []
    for si, st in zip(sinputs, stargets):
        found = True
        for wi in si:
            if wi not in tokens:
                found = False
                break
        for wt in st:
            if wt not in tokens:
                found = False
                break
        if found:
            inputs.append(si)
            targets.append(st)
    return inputs, targets

class Vocab():
    """Default Vocabulary Class"""
    def __init__(self, tokens):
        """Initialize Vocab from a list of TOKENS"""
        self.stokens = ["<START>", "<END>", "<MISSING>", "<PAD>"]
        self.tokens = tokens
        self.ntokens = len(self.tokens) + len(self.stokens)
        self.tokens_to_idx = {token: i for i, token in 
                              enumerate(self.tokens + self.stokens)}
        self.idx_to_tokens = {i: token for token, i in self.tokens_to_idx.items()}

    def get_tokens(self, special_tokens = False):
        return self.tokens + self.stokens if special_tokens else self.tokens

    def get_special_token(self, token='START'):
        try:
            token = "<" + token.upper() + ">"
            return self.id_from_token(token)
        except:
            Warning("Special token {} not found".format(token))
            return None

    def id_from_token(self, token):
        """Return IDX from TOKEN"""
        return (self.tokens_to_idx[token] 
                if token in self.tokens_to_idx.keys() else None)
    
    def token_from_id(self, idx):
        """Return TOKEN from IDX"""
        return (self.idx_to_tokens[idx] 
                if idx in self.idx_to_tokens.keys() else None)
    
    def add_token(self, token):
        """Add TOKEN to TOKENS"""
        self.tokens_to_idx['token'] = self.ntokens
        self.idx_to_tokens[self.ntokens] = token
        self.ntokens += 1
            
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.token_from_id(item)
        elif isinstance(item, slice):
            start = 0 if item.start is None else item.start
            stop = self.ntokens if item.stop is None else item.stop
            step = 1 if item.step is None else item.step
            return [self.token_from_id(i) for i in range(start, stop, step)]
        elif isinstance(item, str):
            return self.id_from_token(item)
        elif isinstance(item, list):
            return [self.id_from_token(i) for i in item]
        else:
            return self.id_from_token["<MISSING>"]

def test():
    DATA_DIR='./data/cornell_movie-dialogs_corpus/'
    MAX_WORD_LEN = 20
    inputs, targets = get_sentences(DATA_DIR, MAX_WORD_LEN)
    tokens = get_tokens_from_data(inputs, targets)
    vocab = Vocab(tokens)
    ti, mi, tt, mt = get_tensors(vocab, [inputs, targets], 
                                      MAX_WORD_LEN)

    return tokens, vocab, inputs, targets, ti, mi, tt, mt

def test_wvectors():
    vectors_file = 'data/glove/glove.6B.50d.txt'
    tokens, vectors = get_tokens_from_vectors_file(vectors_file)
    return tokens, vectors

