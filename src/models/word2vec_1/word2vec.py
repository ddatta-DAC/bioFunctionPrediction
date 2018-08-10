#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "0.0.1"

import gensim
import json
import pandas as pd
import spacy
import textacy
import numpy as np
import pickle
import os
from bioFunctionPrediction.src.utils.dataloader import GODAG
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import operator

# ---Model config--- #
Word_Embed_Size = 512
epochs = 5
# ---------------- #
# MODE = 0 train the model
# MODE = 1 fetch the embedding dict
# ---------------- #
MODE = 1
# this file should have :
# { _id[0...k]: np.array[shape = [Word_Embed_Size]] , ... }

EMDED_FILE = 'GO_word_embed_dict.pkl'
Word2vec_MODEL_FILE = 'word2vec_1.bin'


# ------------------ #
# def get_data():
#     temp_json_2 = 'temp_data_2.json'
#     with open(temp_json_2) as tmp_file:
#         data_dict_2 = json.loads(tmp_file.read())
#     print('Length of dict :: ',len(data_dict_2))
#     return data_dict_2

def get_data():
    df = pd.read_pickle('obo_data.pkl')
    def aux(row):
        return str(row['id']).zfill(7)
    df['id'] = df.apply(aux,axis=1)
    return df


def train():
    global Word_Embed_Size
    global Word2vec_MODEL_FILE
    global epochs
    data_df = get_data()
    sentences = []

    for i, row in data_df.iterrows():
        sentences.append(row['txt'])

    model = gensim.models.Word2Vec(
        sentences,
        iter=epochs,
        window=3,
        size=Word_Embed_Size,
        workers=8,
        min_count=1
    )
    print('Model', model)
    model.save(Word2vec_MODEL_FILE)


def load_model():
    global Word2vec_MODEL_FILE
    # load model
    model = gensim.models.Word2Vec.load(Word2vec_MODEL_FILE)
    return model


def create_embed_dict():
    global Word_Embed_Size
    global MODE

    GODAG_obj = GODAG()
    GODAG_obj.initialize_idmap(None, None)
    idmap = GODAG_obj.idmap

    def _format(k):
        return k.replace('GO:', '')

    idmap = {_format(k): v for k, v in idmap.items()}
    print(' Length of id-map ', len(idmap))
    print('----')

    if MODE == 0:
        train()

    model = load_model()
    emb_dict = {}
    data_df = get_data()
    words = model.wv.vocab
    print('Number of words ', len(words))
    not_found = 0

    for i, row in data_df.iterrows():
        k = row['id'].zfill(7)
        sent_vec = np.zeros([Word_Embed_Size])
        sent = row['txt']
        sent = set(sent)
        if 'OBSOLETE' in sent:
            continue

        for w in sent:
            try:
                vec = np.array(model.wv.word_vec(w))
                sent_vec = sent_vec + vec
            except:
                test = w in words
                print('Word not found ', w, 'in Vocab ', test)

        try:
            key_id = k
            emb_dict[key_id] = sent_vec
        except:
            not_found += 1

    print('Keys not found ... ', not_found)
    return emb_dict


def initialize():
    global EMDED_FILE
    global MODE

    if MODE == 0:
        res = create_embed_dict()
        with open(EMDED_FILE, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif MODE == 1:
        if os.path.isfile(EMDED_FILE):
            with open(EMDED_FILE, 'rb') as handle:
                res = pickle.load(handle)
        else:
            res = create_embed_dict()
    return res


def setup():
    global MODE
    initialize()
    MODE = 1

setup()


# ------------------------ #
# Use this function to extract the embedding dictionary

def get_id_embed_dict():
    return initialize()


# --------------------- #
# TEST MODEL #
# --------------------- #

def test_id(id, embed_dict):

    print('In test_id id :: ', id)
    df = get_data()
    res_dict = OrderedDict()

    id_emb = embed_dict[id]
    id_emb = np.reshape(id_emb, [1, -1])

    # get ids of same type
    _type = list(df.loc[df['id'] == id]['type'])[0]
    id_list = list(df.loc[df['type'] == _type]['id'])

    for candidate in id_list:
        val = np.reshape(embed_dict[candidate], [1, -1])
        sim = cosine_similarity(id_emb, val)[0][0]
        if candidate != id:
            res_dict[candidate] = sim

    sorted_d = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
    res = [item[0] for item in sorted_d[0:5]]

    print('-----------')
    print('Test id ', id)
    print('Top 5', res)
    print('-----------')


test_ids = ['0001234', '0000401', '0031386', '0000730', '0001571', '0002377']
embed_dict = get_id_embed_dict()

for t in test_ids:
    test_id(t, embed_dict)
