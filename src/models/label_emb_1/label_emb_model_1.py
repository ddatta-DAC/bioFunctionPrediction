#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
                *.py: Description of what * does.
                Last Modified:
"""

__author__ = "Debanjan Datta"
__email__ = "ddtta@vt.edu"
__version__ = "0.0.1"
__processor__ = 'label_emb_model_1'

# from models.conv_autoencoder_v1_0 import ConvAutoEncoder

import json
import time
import logging
import os
import pandas as pd
import numpy as np
import glob
import tensorflow as tf
import pickle

try:
    from utils.dataloader import GODAG, FeatureExtractor
    from utils.dataloader import DataIterator, DataLoader
    from predict import predict_evaluate
    from models.conv_autoencoder import ConvAutoEncoder
    from models.conv_autoencoder_v1_0 import ConvAutoEncoder
    from models.label_emb_1.joint_inf_decoder import joint_inf_decoder
except:
    from bioFunctionPrediction.src.utils.dataloader import GODAG, FeatureExtractor
    from bioFunctionPrediction.src.utils.dataloader import DataIterator, DataLoader
    from bioFunctionPrediction.src.predict import predict_evaluate
    from bioFunctionPrediction.src.models.conv_autoencoder_v1_0 import ConvAutoEncoder
    from bioFunctionPrediction.src.models.label_emb_1.joint_inf_decoder import joint_inf_decoder
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')
FLAGS = tf.app.flags.FLAGS
THRESHOLD_RANGE = np.arange(0.1, 0.5, 0.05)


def create_args():

    tf.app.flags.DEFINE_string(
        'resources',
        './data',
        "path to data")

    tf.app.flags.DEFINE_string(
        'outputdir',
        './output',
        "output directory")

    tf.app.flags.DEFINE_string(
        'function',
        'mf',
        'default function to run'
    )

    tf.app.flags.DEFINE_string(
        'inputfile',
        './data',
        "path to data")

    tf.app.flags.DEFINE_integer(
        'trainsize',
        100,
        'number of train batches'
    )
    tf.app.flags.DEFINE_integer(
        'testsize',
        50,
        'number of train batches'
    )
    tf.app.flags.DEFINE_integer(
        'batchsize',
        16,
        'size of batch'
    )
    tf.app.flags.DEFINE_integer(
        'maxseqlen',
        2000,
        'maximum sequence length'
    )
    tf.app.flags.DEFINE_integer(
        'validationsize',
        100,
        'number of validation batches to use'
    )
    tf.app.flags.DEFINE_integer(
        'num_epochs',
        5,
        'number of epochs'
    )
    tf.app.flags.DEFINE_string(
        'pretrained',
        '',
        'location of pretrained embedding'
    )
    tf.app.flags.DEFINE_string(
        'featuretype',
        'ngrams',
        'location of pretrained embedding'
    )

    return


# --------------------- #


# Returns the directory of pretrained model and the name of the meta file
def get_cae_pretrained_dir(model_name_prefix):
    z = glob.glob(os.path.join(FLAGS.outputdir, model_name_prefix + '*'))

    if len(z) > 0:
        print('Pretrained model exists!')
        log.info('Pretrained model exists!')
        z = list(sorted(z))
        # Choose the latest model
        target_dir = z[-1]
        m_files = glob.glob(os.path.join(target_dir, '*.meta'))
        try:
            meta_file = m_files[0]
        except:
            meta_file = None
        return target_dir, meta_file

    return None, None


def get_iterators():

    funcs = pd.read_pickle(os.path.join(
        FLAGS.resources,
        '{}.pkl'.format((FLAGS.function).lower())
    ))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)

    data = DataLoader(filename=FLAGS.inputfile)

    train_iter = DataIterator(
        batchsize=FLAGS.batchsize,
        size=FLAGS.trainsize,
        seqlen=FLAGS.maxseqlen,
        dataloader=data,
        filename='train',
        filterByEvidenceCodes=True,
        functype=FLAGS.function,
        featuretype='ngrams',
        onlyLeafNodes =True
    )

    valid_iter = DataIterator(
        batchsize=FLAGS.batchsize,
        size=FLAGS.validationsize,
        seqlen=FLAGS.maxseqlen,
        dataloader=data,
        functype=FLAGS.function,
        featuretype='ngrams',
        filename='validation',
        filterByEvidenceCodes=True,
        onlyLeafNodes=True
    )

    test_iter = DataIterator(
        batchsize=FLAGS.batchsize,
        size=FLAGS.testsize,
        seqlen=FLAGS.maxseqlen,
        dataloader=data,
        functype=FLAGS.function,
        featuretype='ngrams',
        filename='test',
        filterByEvidenceCodes=True,
        onlyLeafNodes=True
    )
    data.close()
    return train_iter, test_iter, valid_iter


def pretrain_cae_model(model_savename):
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)

    train_iter, test_iter, valid_iter = get_iterators()
    log.info('using feature type - {}'.format(FLAGS.featuretype))

    if FLAGS.featuretype == 'onehot':
        vocabsize = len(FeatureExtractor.aminoacidmap)
    else:
        vocabsize = len(FeatureExtractor.ngrammap)

        # Build graph
    cae_model_obj = ConvAutoEncoder(
        vocab_size=vocabsize,
        maxlen=train_iter.expectedshape,
        batch_size=FLAGS.batchsize,
        embedding_dim=512
    )
    cae_model_obj.build()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    init.run(session=sess)
    chkpt = tf.train.Saver(max_to_keep=5)

    train_writer = tf.summary.FileWriter(
        FLAGS.outputdir + '/train',
        sess.graph
    )
    test_writer = tf.summary.FileWriter(
        FLAGS.outputdir + '/test'
    )

    metagraphFlag = True
    log.info('Starting epochs')

    # -------------------------- #
    #  Start training the autoencoder
    # -------------------------- #
    bestloss = np.infty
    maxwait = 10
    wait = 0
    slack = 1e-5
    step = 0
    earlystop = False
    for epoch in range(FLAGS.num_epochs):
        for x, y in train_iter:
            if x.shape[0] != y.shape[0]:
                raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))

            _, loss = sess.run([
                cae_model_obj.train,
                cae_model_obj.loss
            ],
                feed_dict={
                    cae_model_obj.x_input: x
                }
            )
            log.info('step :: {}, loss :: {}'.format(step, np.round(loss, 3)))
            step += 1
            if step % 100 == 0:
                x, y = valid_iter.__next__()
                valid_loss, tp = sess.run(
                    [cae_model_obj.loss, cae_model_obj.truepos],
                    feed_dict={cae_model_obj.x_input: x}
                )
                log.info('Validation loss at step: {} is {} '.format(step, np.round(valid_loss, 3)))

                if (valid_loss <= (bestloss + slack)) or (valid_loss + slack <= bestloss):
                    wait = 0
                    bestloss = valid_loss
                    chkpt.save(
                        sess,
                        os.path.join(
                            FLAGS.outputdir,
                            model_savename,
                            'model_epoch{}'.format(epoch)
                        ),
                        global_step=step,
                        write_meta_graph=metagraphFlag
                    )
                else:
                    wait += 1
                    if wait > maxwait:
                        earlystop = True
                        break

        chkpt.save(
            sess,
            os.path.join(
                FLAGS.outputdir,
                model_savename,
                'model_epoch{}'.format(epoch)
            ),
            global_step=step,
            write_meta_graph=metagraphFlag
        )
        train_iter.reset()
        valid_iter.reset()
        if earlystop:
            log.info('stopping early at epoch: {}, step:{}, loss:{}'.format(epoch, step, bestloss))
            break
    sess.close()
    tf.reset_default_graph()


def get_pretrained_cae():
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)

    model_name_prefix = 'label_emb_1_cae'
    model_savename = '{}_{}_{}'.format(model_name_prefix, __processor__, int(time.time()))
    pretrained_dir, meta_file = get_cae_pretrained_dir(model_name_prefix)

    # ------------------- #
    # Pre-train the model #
    # ------------------- #
    if pretrained_dir is None:
        pretrain_cae_model(model_savename)

    if FLAGS.featuretype == 'onehot':
        vocabsize = len(FeatureExtractor.aminoacidmap)
    else:
        vocabsize = len(FeatureExtractor.ngrammap)

    train_iter, _, _ = get_iterators()

    # Build graph
    cae_model_obj = ConvAutoEncoder(
        vocab_size=vocabsize,
        maxlen=train_iter.expectedshape,
        batch_size=FLAGS.batchsize,
        embedding_dim=512
    )
    cae_model_obj.build(pretrained_dir)
    pretrained_dir, meta_file = get_cae_pretrained_dir(model_name_prefix)

    x_inp = None
    encoder_op = None
    tf.reset_default_graph()
    sess = tf.Session()

    if pretrained_dir is not None:

        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(pretrained_dir))

        graph = tf.get_default_graph()
        names = cae_model_obj.return_wt_list_to_restore()

        for n in names[1:]:
            try:
                graph.get_tensor_by_name(n)
            except:
                print(' Error restoring ..', n)

        x_inp = graph.get_tensor_by_name("model_input/x:0")
        encoder_op = graph.get_tensor_by_name("Encoder/Relu_4:0")

    return sess, cae_model_obj, x_inp, encoder_op


# ------------------------- #
# Validate the final model
# ------------------------- #
def validate(dataiter, sess, x_inp, decoder, summary_writer):
    step = 0
    avgPrec, avgRecall, avgF1 = (np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE)
                                 )
    for x, y in dataiter:
        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f, summary = sess.run([decoder.precision, decoder.recall,
                                         decoder.f1score, decoder.summary],
                                        feed_dict={decoder.ys_: y,
                                                   x_inp: x,
                                                   decoder.threshold: [thres]})
            summary_writer.add_summary(summary, step)
            prec.append(p)
            recall.append(r)
            f1.append(f)

        avgPrec += prec
        avgRecall += recall
        avgF1 += f1
        step += 1

    dataiter.reset()
    return (avgPrec / step, avgRecall / step, avgF1 / step)

def test(dataiter, sess, x_inp, decoder, summary_writer):
    step = 0
    THRESHOLD_RANGE = np.arange(0.1, 0.5, 0.05)
    avgPrec, avgRecall, avgF1 = (np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE)
                                 )
    for x, y in dataiter:
        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f, summary = sess.run([decoder.precision, decoder.recall,
                                         decoder.f1score, decoder.summary],
                                        feed_dict={decoder.ys_: y,
                                                   x_inp: x,
                                                   decoder.threshold: [thres]})
            summary_writer.add_summary(summary, step)
            prec.append(p)
            recall.append(r)
            f1.append(f)

        avgPrec += prec
        avgRecall += recall
        avgF1 += f1
        step += 1

    dataiter.reset()
    return (avgPrec / step), (avgRecall / step), (avgF1 / step)

def get_word2vec_emb():

    file_loc = './../word2vec_1'
    file_name = 'GO_word_embed_dict.pkl'
    with open(os.path.join(file_loc,file_name), 'rb') as file_handle:
        emb_dict = pickle.load(file_handle)
        return emb_dict


# TODO : FIX this function!!
def create_label_emb_lookup():
    print ( 'create_label_emb_lookup  .... ')

    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    print('1st funcs', funcs ,len(funcs))
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)

    w2v_emb_size = 512
    w2v_emb_dict = get_word2vec_emb()
    goids = GODAG.GOIDS


    # format the keys to add 'GO:'
    w2v_emb_dict = {
        'GO:' + k : v for k,v in w2v_emb_dict.items()
    }


    # 1 more since 0 means unknown
    emb_map = np.zeros([len(funcs)+1,w2v_emb_size])
    for k, emb in w2v_emb_dict.items():
        try:
            idx = goids.index(k)+1
            emb_map[idx] = emb
        except :
            pass
    print (emb_map.shape)
    return emb_map


def joint_inf_layer():



    return


def build_model():

    w2v_emb_dim = 512
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)


    train_iter, test_iter, valid_iter = get_iterators()
    sess, cae_model_obj, x_inp, encoder_op = get_pretrained_cae()

    label_emb = create_label_emb_lookup()
    label_count = len(funcs)
    x_shape = encoder_op.shape.as_list()[1:]
    print(encoder_op.shape.as_list()[1:])

    x,y = train_iter.__next__()
    print(x.shape)
    print(y.shape)

    y_shape = [label_count,w2v_emb_dim]
    decoder = joint_inf_decoder(
        x_shape,
        y_shape,
        label_count,
        w2v_emb_dim
    )

    print('y ' , y )
    decoder.build()
    return


def main(argv):
    build_model()

if __name__ == "__main__":
    create_args()
    tf.app.run(main)


