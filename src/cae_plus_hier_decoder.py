#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "0.0.1"
__processor__ = 'deepGO'

import time
import pandas as pd
import tensorflow as tf

# from models.conv_autoencoder_v1_0 import ConvAutoEncoder

import json
import logging
import os
import numpy as np
import glob

try:
    from utils.dataloader import GODAG, FeatureExtractor
    from utils.dataloader import DataIterator, DataLoader
    from predict import predict_evaluate
    from models.conv_autoencoder import ConvAutoEncoder
    from models.decoders import HierarchicalGODecoder
    from models.conv_autoencoder_v1_0 import ConvAutoEncoder
except:
    from bioFunctionPrediction.src.utils.dataloader import GODAG, FeatureExtractor
    from bioFunctionPrediction.src.utils.dataloader import DataIterator, DataLoader
    from bioFunctionPrediction.src.predict import predict_evaluate
    from bioFunctionPrediction.src.models.conv_autoencoder_v1_0 import ConvAutoEncoder
    from bioFunctionPrediction.src.models.decoders import HierarchicalGODecoder

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
        'inputfile',
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


# ----------------------------------------- #
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
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
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
        featuretype='ngrams'
    )

    valid_iter = DataIterator(
        batchsize=FLAGS.batchsize,
        size=FLAGS.validationsize,
        seqlen=FLAGS.maxseqlen,
        dataloader=data,
        functype=FLAGS.function,
        featuretype='ngrams',
        filename='validation',
        filterByEvidenceCodes=True
    )

    test_iter = DataIterator(
        batchsize=FLAGS.batchsize,
        size=FLAGS.testsize,
        seqlen=FLAGS.maxseqlen,
        dataloader=data,
        functype=FLAGS.function,
        featuretype='ngrams',
        filename='test',
        filterByEvidenceCodes=True
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

    model_name_prefix = 'cae_pretrain'
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


def build_model():
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)
    train_iter, test_iter, valid_iter = get_iterators()
    sess, cae_model_obj, x_inp, encoder_op = get_pretrained_cae()

    decoder = HierarchicalGODecoder(funcs,
                                    encoder_op,
                                    root=FLAGS.function,
                                    ).build(GODAG)

    init = tf.global_variables_initializer()
    init.run(session=sess)

    chkpt = tf.train.Saver(max_to_keep=4)
    train_writer = tf.summary.FileWriter(FLAGS.outputdir + '/train',
                                         sess.graph)

    test_writer = tf.summary.FileWriter(FLAGS.outputdir + '/test')
    step = 0
    maxwait = 5
    wait = 0
    best_f1 = 0
    best_thres = 0
    metagraphFlag = False
    log.info('starting epochs')
    model_savename = 'cae_plus_hier_decoder_{}'.format(int(time.time()))
    meta_graph_file_name = os.path.join(
        FLAGS.outputdir,
        model_savename,
        'model_{}.meta'.format(
            FLAGS.function
        )
    )
    tf.train.export_meta_graph(filename=meta_graph_file_name)
    for epoch in range(FLAGS.num_epochs):
        for x, y in train_iter:
            if x.shape[0] != y.shape[0]:
                raise Exception('invalid, x :: {}, '
                                'y :: {}'.format(str(x.shape), str(y.shape)))

            _, total_loss, summary = sess.run(
                [decoder.train, decoder.loss, decoder.summary],
                feed_dict={
                    x_inp: x,
                    decoder.ys_: y,
                    decoder.threshold: [.3]
                }
            )
            train_writer.add_summary(summary, step)
            log.info(
                'step :: {}, total_loss :: {} '.format(
                    step, np.round(total_loss, 3)))
            step += 1

            if step % (5) == 0 :
                log.info('beginning validation')
                prec, recall, f1 = validate(
                    valid_iter,
                    sess,
                    x_inp,
                    decoder,
                    test_writer
                )
                thres = np.argmax(np.round(f1, 3))
                log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(
                    epoch,
                    np.round(prec, 2)[thres],
                    np.round(recall, 2)[thres],
                    np.round(f1, 2)[thres])
                )
                log.info('selected threshold is {}'.format(thres / 10 + 0.1))
                if f1[thres] > (best_f1 + 1e-3):
                    best_f1 = f1[thres]
                    best_thres = THRESHOLD_RANGE[thres]
                    wait = 0
                    chkpt.save(
                        sess,
                        os.path.join(
                            FLAGS.outputdir,
                            model_savename,
                            'model_{}_{}'.format(FLAGS.function, step)
                        ),
                        global_step=step,
                        write_meta_graph=metagraphFlag
                    )
                    metagraphFlag = False
                else:
                    wait += 1
                    if wait > maxwait:
                        log.info('f1 didnt improve for last {} validation steps, so stopping'.format(maxwait))
                        break

        train_iter.reset()

        log.info('testing model')
        placeholders = [x_inp.name, 'y_out:0', 'thres:0']
        prec, recall, f1 = predict_evaluate(
            test_iter,
            [best_thres],
            placeholders,
            os.path.join(FLAGS.outputdir, model_savename)
        )
        log.info('test results')
        log.info('precision: {}, recall: {}, F1: {}'.format(round(prec, 3), round(recall, 3), round(f1, 3)))

def main(argv):
    # pretrain the conv auto encoder
    build_model()


if __name__ == "__main__":
    create_args()
    tf.app.run(main)
