# -*- coding:utf-8 -*-
__author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "0.0.1"

import glob
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
import os
import numpy as np
import json
import logging
import tarfile
import wget
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger('root.DataLoader')

'''
We ignore 5 amino acids. Rest 20(k) form our vocab size.
These are delegated to the id  k+1
0 is used for padding.
'''

# --------------------------------- #
# Set up the data
# Done only once
# After the files are created (locally), this will be avoided
# Checks put in place
# --------------------------------- #

IGNORE_AA = ('B', 'O', 'J', 'X', 'U', 'Z')
UNKNOWN_AA = '_'
CAT_GO = ('BP', 'MF')
ORIG_SETS = ('test', 'train')
# ---- #
# Effective  Test size : 0.15
# Effective validation size : 0.05 (0.25 * 0.20)

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25

file_name_dict_y = {
    'BP': 'bp.pkl',
    'MF': 'mf.pkl'
}

file_name_dict_x = {
    'BP': {
        'train': 'train-bp.pkl',
        'test': 'test-bp.pkl'
    },
    'MF': {
        'train': 'train-mf.pkl',
        'test': 'test-mf.pkl'
    }
}

orig_file_dir = os.path.join(os.path.dirname(__file__), './../../resources/data/data_orig_paper/train')
data_dir = os.path.join(os.path.dirname(__file__), './../../resources/data/data_paper')


def download_data():
    disp_msg = 'In download_data'
    log.info(disp_msg)
    print(disp_msg)
    global orig_file_dir
    if not os.path.isdir(orig_file_dir):
        os.mkdir(orig_file_dir)
    else:
        return
    os.chdir(orig_file_dir)
    urls = [
        'http://deepgo.bio2vec.net/data/train.tar.gz',
        'http://deepgo.bio2vec.net/data/test.fa'
    ]

    for url in urls:
        fname = wget.download(url)
        if (fname.endswith("tar.gz")):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
    return


def get_file(cat_go, _set='train'):
    global orig_file_dir
    global orig_file_name_dict_x

    f_name = file_name_dict_x[cat_go][_set]
    file_path = orig_file_dir + '/' + f_name
    data1 = pd.read_pickle(file_path)
    return data1


'''
The original paper uses train and test sets.
We combine these and set up train , test, validations sets
'''

def combine_data():
    disp_msg = 'In combine_data'
    log.info(disp_msg)
    print(disp_msg)

    global CAT_GO
    global ORIG_SETS
    global data_dir

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    for _go in CAT_GO:
        # create a new dataframe
        df = pd.DataFrame(
            columns=['sequences', 'ngrams', 'labels']
        )
        for _set in ORIG_SETS:
            data = get_file(_go, _set)
            data = data[['sequences', 'ngrams', 'labels']]
            df = df.append(data, ignore_index=True, sort=True)

        train, test = train_test_split(df, test_size=0.2)
        test, val = train_test_split(test, test_size=0.25)
        _ref = {}
        _ref['train'] = train
        _ref['test'] = test
        _ref['val'] = val

        for k, _df in _ref.items():
            _df = _df.reset_index()
            file_name = ''.join([_go.upper(), '_', k, '.pkl'])
            res_path = os.path.join(data_dir, file_name)
            _df.to_pickle(res_path)


def initialize():
    global orig_file_dir
    global data_dir

    if not os.path.isdir(orig_file_dir):
        download_data()
    if not os.path.isdir(data_dir):
        combine_data()
    return


initialize()


# ---------------------------------------- #

class Amino_Acid_Map:

    def __init__(self):
        self.aminoacid_map = {}
        self.load()
        return

    def load(self):
        global IGNORE_AA
        data_dir = os.path.join(os.path.dirname(__file__), './../../resources')

        with open(os.path.join(data_dir, 'aminoacids.txt'), 'r') as inpf:
            _id = 1
            aa_list = list(set(json.load(inpf)))

            for aa in aa_list:
                if aa not in IGNORE_AA:
                    self.aminoacid_map[aa] = _id
                    _id += 1

            aa_size = len(self.aminoacid_map)
            self.aminoacid_map['<unk>'] = aa_size + 1

            log.info(
                'loaded amino acid map of size-{}'.format(aa_size)
            )

    def amino_acid_to_id(self, amino_acid):
        if amino_acid not in self.aminoacid_map:
            log.info('unable to find {} in known aminoacids'.format(amino_acid))
            amino_acid = '<unk>'
        return self.aminoacid_map[amino_acid]

    def to_onehot(self, seq):
        res = [
            self.amino_acid_to_id(amino_acid) for amino_acid in seq
        ]
        return res


# ------------------------- #
class Ngram_Map:

    def __init__(self):
        self.ngram_len = 3
        self.ngram_map = None
        self.load()
        return

    def load(self):
        global IGNORE_AA
        global UNKNOWN_AA
        ngrams_file = 'ngrams_21_aa.json'

        ngrams_file_path = os.path.join(os.path.join(os.path.dirname(__file__), './../../resources'), ngrams_file)
        if os.path.isfile(ngrams_file_path):
            with open(ngrams_file_path, 'r') as file_handle:
                json_str = file_handle.read()
                self.ngram_map = json.loads(json_str)
        else:
            data_dir = os.path.join(os.path.dirname(__file__), './../../resources')
            inp_file = os.path.join(data_dir, 'ngrams.txt')
            file_handle = open(inp_file, 'r')
            ngrams = json.load(file_handle)

            # Replace ignored char with _
            ngrams_list = []
            for _ngram in ngrams:
                for _ignore_aa in IGNORE_AA:
                    _n = _ngram.replace(_ignore_aa, UNKNOWN_AA)
                    ngrams_list.append(_n)

            self.ngrams_list = set(ngrams_list)

            self.ngram_map = {}
            log.info(
                'loaded amino acid ngram list of size-{}'.format(len(self.ngrams_list))
            )
            id = 1
            for ng in self.ngrams_list:
                self.ngram_map[ng] = id
                id += 1

            if '___' not in self.ngram_map.keys():
                self.ngram_map['___'] = id
            log.info(
                'loaded amino acid ngram map of size-{}'.format(len(self.ngram_map))
            )

            data_dir = os.path.join(os.path.dirname(__file__), './../../resources')
            inp_file = os.path.join(data_dir, ngrams_file)
            json_data = json.dumps(self.ngram_map)
            file_handle = open(inp_file, 'w')
            file_handle.write(json_data)
            file_handle.close()
        return

    def aa_ngram_to_id(self, amino_acid):
        if amino_acid not in self.ngram_map:
            log.info('unable to find {} in known aminoacids'.format(amino_acid))
            amino_acid = '___'
        return self.ngram_map[amino_acid]

    def to_ngram(self, seq):
        res = []
        for ig in IGNORE_AA:
            seq = seq.replace(ig, '_')

        for i in range(len(seq) - self.ngram_len):
            subseq = seq[i:i + self.ngram_len]
            res.append(self.aa_ngram_to_id(subseq))
        return res


# ------------------------- #

class BaseDataIterator:
    global data_dir
    data_loc = os.path.join(os.path.dirname(__file__), data_dir)

    def __init__(
            self,
            functype,
            batch_size,
            max_batch_count = None,
            seqlen=200,
            featuretype='onehot',
            autoreset=False,
    ):
        self.featuretype = featuretype
        self.functype = functype.upper()
        self.max_seq_len = seqlen
        self.seq_col_name = 'sequences'
        self.batch_size = batch_size
        self.aa_map_obj = Amino_Acid_Map()
        self.ngram_map_obj = Ngram_Map()
        self.y_column = 'labels'
        self.autoreset = autoreset
        self.max_batch_count = max_batch_count
        self.batch_count = 0
        self.reset()
        self.ngram_size = 3
        self.expectedshape = ((self.max_seq_len - self.ngram_size + 1) if self.featuretype == 'ngrams' else self.max_seq_len)
        self.x_column = 'sequences'
        self.input_column = 'x'
        print('Max Seq len' , self.max_seq_len)
        print('Expected shape', self.expectedshape , 'x_column', self.x_column)
        if self.featuretype == 'onehot':
            self.vocab_size = len(self.aa_map_obj.aminoacid_map)
        else :
            self.vocab_size = len(self.ngram_map_obj.ngram_map)
        return

    def __iter__(self):
        return self

    def reset(self):
        # start of data file
        self.cur_idx = 0
        self.batch_count = 0

    def filter_by_seq_len(self):
        self.df = self.df[self.df[self.seq_col_name].str.len() <= self.max_seq_len]
        return

    def convert_seq_to_id(self):
        def pad_seq_1(res):
            pad = [0] * (self.max_seq_len - len(res))
            res.extend(pad)
            return res

        def aux_1(row):
            seq = row[self.x_column]
            res = self.aa_map_obj.to_onehot(seq)
            return pad_seq_1(res)
        
        _df = pd.DataFrame(self.df,copy=True)
        _df['x'] = self.df.apply(aux_1, axis=1)
        return _df

    def convert_seq_to_ngram_id(self):

        def pad_seq_2(res):
            pad = [0] * (self.expectedshape - len(res))
            res.extend(pad)
            return res

        def aux_2(row):
            seq = row[self.x_column]
            res = self.ngram_map_obj.to_ngram(seq)
            res = pad_seq_2(res)
            return res

        _df = pd.DataFrame(self.df,copy=True)
        _df['x'] = _df.apply(aux_2, axis=1)
        return _df

    def format_x(self):
        if self.featuretype == 'onehot':
            _df = self.convert_seq_to_id()
        elif self.featuretype == 'ngrams':
            _df = self.convert_seq_to_ngram_id()
        else:
            log.error( 'Wrong Feature type :: ' + self.featuretype)
            exit(2) 
        try:
            del _df[self.x_column]
            del _df['ngrams']
        except:
            pass
        return _df

    # def convert_to_1hot(self, _x):
    #     res = []
    #     for _xi in _x :
    #         onehot_enc = OneHotEncoder(n_values=self.vocab_size)
    #         print(onehot_enc.transform(np.reshape(_x,[1,-1])))
    #         exit(1)
    #         # print(tmp.shape)
    #         # res.append(tmp)
    #     res = np.array(res)
    #     print(res.shape)
    #     return res


    def format_batch_data(self, _df):
        y = list(_df[self.y_column])
        y = np.asarray(y)
        x_data = _df[self.input_column].values
        x = np.hstack([np.array(i) for i in x_data])
        x = np.reshape(x, [self.batch_size, -1])
        return x, y

    def generate_batch(self):
        log.info(
            'Data Iterator object geenrating batch of size :: {}'.format(self.batch_size)
        )

        start_idx = self.cur_idx
        end_idx = self.cur_idx + self.batch_size - 1
        if (self.batch_count + 1) > self.max_batch_count:
            if self.autoreset:
                self.reset()
                start_idx = self.cur_idx
                end_idx = self.cur_idx + self.batch_size - 1
            else:
                raise StopIteration

        self.batch_count += 1
        tmp_df = self.df.loc[start_idx:end_idx]
        tmp_df = pd.DataFrame(tmp_df, copy=True)
        return self.format_batch_data(tmp_df)

    def __next__(self):
        return self.generate_batch()

    def set_batch_limit(self):
        if self.max_batch_count == None:
            self.max_batch_count = len(self.df)/self.batch_size

class TrainIterator(BaseDataIterator):
    file_path = None

    def __init__(
            self,
            functype,
            batch_size,
            max_batch_count = None,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        BaseDataIterator.__init__(
            self,
            functype,
            batch_size,
            max_batch_count,
            seqlen,
            featuretype,
            autoreset
        )
        self.read_data()
        self.df = self.format_x()
        self.set_batch_limit()
        return

    def read_data(self):
        file_name = ''.join([str(self.functype), '_', 'train.pkl'])
        TrainIterator.file_path = os.path.join(BaseDataIterator.data_loc, file_name)
        self.df = pd.read_pickle(TrainIterator.file_path)
        self.filter_by_seq_len()
        log.info(
            'Train Data size :: {}'.format(len(self.df))
        )
        return
		
class TestIterator(BaseDataIterator):
    file_path = None

    def __init__(
            self,
            functype,
            batch_size,
            max_batch_count=None,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        BaseDataIterator.__init__(
            self,
            functype,
            batch_size,
            max_batch_count,
            seqlen,
            featuretype,
            autoreset)
        self.read_data()
        self.df = self.format_x()
        self.set_batch_limit()
        return

    def read_data(self):
        file_name = ''.join([str(self.functype), '_', 'test.pkl'])
        TestIterator.file_path = os.path.join(BaseDataIterator.data_loc, file_name)
        self.df = pd.read_pickle(TestIterator.file_path)

        # filter data by sequence length
        self.filter_by_seq_len()
        log.info(
            'Test Data size :: {}'.format(len(self.df))
        )


class ValidIterator(BaseDataIterator):
    file_path = None

    def __init__(
            self,
            functype,
            batch_size,
            max_batch_count=None,
            seqlen=2000,
            featuretype='onehot',
            autoreset=False,
    ):
        BaseDataIterator.__init__(
            self,
            functype,
            batch_size,
            max_batch_count,
            seqlen,
            featuretype,
            autoreset)
        self.read_data()
        self.df = self.format_x()
        self.set_batch_limit()
        return

    def read_data(self):
        file_name = ''.join([str(self.functype), '_', 'val.pkl'])
        ValidIterator.file_path = os.path.join(BaseDataIterator.data_loc, file_name)
        self.df = pd.read_pickle(ValidIterator.file_path)
        self.filter_by_seq_len()
        log.info(
            'Validation Data size :: {}'.format(len(self.df))
        )


def functional_test():
    ti = TrainIterator('MF', 256, featuretype='ngrams')
    x, y = ti.__next__()
    print(x.shape, y.shape)
    x, y = ti.__next__()
    print(x.shape, y.shape)

    ti = ValidIterator('MF', 384, featuretype='ngrams')
    x, y = ti.__next__()
    print(x.shape, y.shape)
    x, y = ti.__next__()
    print(x.shape, y.shape)

    ti = TestIterator('BP', 32, featuretype='ngrams')
    x, y = ti.__next__()
    print(x.shape, y.shape)
    x, y = ti.__next__()
    print(x.shape, y.shape)


def iter_test():
    ti = TrainIterator('MF', 256, featuretype='ngrams',max_batch_count = 100,seqlen=2002)
    for x, y in ti:
        print(x.shape, y.shape)
        n_values = len(ti.ngram_map_obj.ngram_map) + 1
        print (x[10])
        print(np.eye(n_values)[x[10]])
        exit(1)

# ----- #


_Amino_Acid_Map = Amino_Acid_Map()
_Ngram_Map = Ngram_Map()

# iter_test()
# functional_test()
