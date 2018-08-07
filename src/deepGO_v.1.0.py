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
__version__ = "0.1.0"
__processor__ = 'deepGO'

import time
import pandas as pd
import tensorflow as tf
import json
import logging
import os
import numpy as np
#import ipdb

try:
    import utils.experimental_datareader as dataloader
    from utils.experimental_datareader import TrainIterator, TestIterator, ValidIterator
    from models.encoders import CNNEncoder
    from models.decoders import HierarchicalGODecoder
    from predict import predict_evaluate
    import utils.experimental_datareader as new_dataloader
    from utils.dataloader import GODAG
    from utils.dataloader import FeatureExtractor
    import utils.dataloader as old_dataloader
    import utils
except:
    import bioFunctionPrediction.src.utils.experimental_datareader as dataloader
    from bioFunctionPrediction.src.models.encoders import CNNEncoder
    from bioFunctionPrediction.src.models.decoders import HierarchicalGODecoder
    from bioFunctionPrediction.src.predict import predict_evaluate
    import bioFunctionPrediction.src.utils.experimental_datareader as new_dataloader
    from bioFunctionPrediction.src.utils.experimental_datareader import TrainIterator, TestIterator, ValidIterator
    import bioFunctionPrediction.src.utils.dataloader as old_dataloader
    from bioFunctionPrediction.src.utils.dataloader import GODAG
    from bioFunctionPrediction.src.utils.dataloader import FeatureExtractor
    import bioFunctionPrediction.src.utils as utils
# ------------------------- #

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
        20,
        'number of train batches'
    )

    tf.app.flags.DEFINE_integer(
        'testsize',
        2,
        'number of train batches'
    )

    tf.app.flags.DEFINE_integer(
        'validsize',
        2,
        'number of train batches'
    )

    tf.app.flags.DEFINE_integer(
        'batchsize',
        16,
        'size of batch'
    )

    tf.app.flags.DEFINE_integer(
        'maxseqlen',
        2002,
        'maximum sequence length'
    )

    tf.app.flags.DEFINE_integer(
        'validationsize',
        100,
        'Number of validation batches to use'
    )

    tf.app.flags.DEFINE_integer(
        'num_epochs',
        2,
        'number of epochs'
    )

    tf.app.flags.DEFINE_string(
        'featuretype',
        'ngrams',
        "Ngrams or one hot encoded input")

    tf.app.flags.DEFINE_string(
        'pretrained',
        '',
        'location of pretrained embedding'
    )
    return


# ---------------------------------- #

def validate(
        dataiter,
        sess,
        encoder,
        decoder,
        summary_writer
):
    step = 0
    avgPrec, avgRecall, avgF1 = (
        np.zeros_like(THRESHOLD_RANGE),
        np.zeros_like(THRESHOLD_RANGE),
        np.zeros_like(THRESHOLD_RANGE)
    )

    for x, y in dataiter:
        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f, summary = sess.run([
                decoder.precision,
                decoder.recall,
                decoder.f1score,
                decoder.summary
            ],
            feed_dict={
                decoder.ys_: y,
                encoder.xs_: x,
                decoder.threshold: [thres]
            }
            )
            summary_writer.add_summary(summary, step)
            prec.append(p)
            recall.append(r)
            f1.append(f)

        avgPrec += prec
        avgRecall += recall
        avgF1 += f1
        step += 1

    log.info('finished evaluating {} validation steps'.format(step))
    dataiter.reset()
    return (avgPrec / step, avgRecall / step, avgF1 / step)


def main(argv):

    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    _ = GODAG.initialize_idmap(funcs, FLAGS.function)
    

    log.info('GO DAG initialized. Updated function list :: {}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')

    # data = DataLoader(filename=FLAGS.inputfile)
    train_iter = TrainIterator(
        batch_size=FLAGS.batchsize,
        max_batch_count=FLAGS.trainsize,
        seqlen=FLAGS.maxseqlen,
        functype=FLAGS.function,
        featuretype=FLAGS.featuretype
    )

    valid_iter = ValidIterator(
        batch_size=FLAGS.batchsize,
        max_batch_count=FLAGS.validsize,
        seqlen=FLAGS.maxseqlen,
        functype=FLAGS.function,
        featuretype=FLAGS.featuretype
    )

    test_iter = ValidIterator(
        batch_size=FLAGS.batchsize,
        max_batch_count=FLAGS.testsize,
        seqlen=FLAGS.maxseqlen,
        functype=FLAGS.function,
        featuretype=FLAGS.featuretype
    )

    modelsavename = 'deepgo_v010_{}'.format(int(time.time()))
    pretrained = None

    # if FLAGS.pretrained != '':
    #     pretrained, ngrammap = utils.load_pretrained_embedding(FLAGS.pretrained)
    #     FeatureExtractor.ngrammap = ngrammap
    if FLAGS.featuretype == "onehot" :
        vocab_size =  len(new_dataloader._Amino_Acid_Map.aminoacid_map)
    else :
        vocab_size = len(new_dataloader._Ngram_Map.ngram_map)

    with tf.Session() as sess:
        log.info('--- Satrting session ---')
        encoder = CNNEncoder(
            vocab_size=vocab_size,
            inputsize=train_iter.expectedshape,
            pretrained_embedding=pretrained
        ).build()

        log.info('Built encoder')
        decoder = HierarchicalGODecoder(
            funcs,
            encoder.outputs,
            root=FLAGS.function
        ).build(GODAG)

        log.info('Built decoder')

        init = tf.global_variables_initializer()
        init.run(session=sess)
        chkpt = tf.train.Saver(max_to_keep=4)
        train_writer = tf.summary.FileWriter(FLAGS.outputdir + '/train', sess.graph)

        test_writer = tf.summary.FileWriter(FLAGS.outputdir + '/test')
        step = -1
        maxwait = 5
        wait = 0
        bestf1 = 0
        bestthres = 0
        metagraphFlag = True
        log.info('starting epochs')
        tf.train.export_meta_graph(filename=os.path.join(FLAGS.outputdir, modelsavename,
                                                         'model_{}.meta'.format(FLAGS.function)))
        for epoch in range(FLAGS.num_epochs):
            log.info('************EPOCH-{} *******'.format(epoch))
            for x, y in train_iter:
                if x.shape[0] != y.shape[0]:
                    raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))
               

                _, loss, summary = sess.run(
                    [decoder.train, decoder.loss, decoder.summary],
                    feed_dict={
                        encoder.xs_: x,
                        decoder.ys_: y,
                        decoder.threshold: [.3]
                    }
                )


                train_writer.add_summary(summary, step)
                log.info('step-{}, loss-{}'.format(step, round(loss, 2)))
                step += 1
                if step % (100) == 0:
                    log.info('beginning validation')
                    prec, recall, f1 = validate(valid_iter, sess, encoder, decoder, test_writer)
                    thres = np.argmax(np.round(f1, 2))
                    log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(epoch,
                                                                                     np.round(prec, 2)[thres],
                                                                                     np.round(recall, 2)[thres],
                                                                                     np.round(f1, 2)[thres]))
                    log.info('selected threshold is {}'.format(THRESHOLD_RANGE[thres]))
                    if f1[thres] > (bestf1 + 1e-3):
                        bestf1 = f1[thres]
                        bestthres = THRESHOLD_RANGE[thres]
                        wait = 0
                        chkpt.save(sess, os.path.join(FLAGS.outputdir,
                                                      modelsavename,
                                                      'model_{}_{}'.format(FLAGS.function, step)),
                                   global_step=step, write_meta_graph=metagraphFlag)
                        metagraphFlag = False
                    else:
                        wait += 1
                        if wait > maxwait:
                            log.info('f1 didnt improve for last {} validation steps, so stopping'.format(maxwait))
                            break

            train_iter.reset()

    log.info('testing model')
    # test_dataiter = new_dataloader.TestIterator(
    #     functype=FLAGS.function, batch_size=FLAGS.batchsize,
    #                                             featuretype='ngrams', seqlen=FLAGS.maxseqlen,
    #                                             max_batch_count=FLAGS.testsize)

    placeholders = ['x_in:0', 'y_out:0', 'thres:0']
    prec, recall, f1 = predict_evaluate(test_iter, [bestthres], placeholders,
                                        os.path.join(FLAGS.outputdir, modelsavename))
    log.info('test results')
    log.info('precision: {}, recall: {}, F1: {}'.format(np.round(prec, 3), np.round(recall, 3), np.round(f1, 3)))


if __name__ == "__main__":
    create_args()
    tf.app.run(main)
