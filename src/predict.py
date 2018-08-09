#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"
__processor__ = 'deepGO'

import time
import numpy as np
import pandas as pd
import tensorflow as tf
# from models.encoders import CNNEncoder
# from models.decoders import HierarchicalGODecoder
import json
import logging
import os
from glob import glob

try:
    from utils.dataloader import GODAG, FeatureExtractor
    from utils.dataloader import DataIterator, DataLoader
except:
    from bioFunctionPrediction.src.utils.dataloader import GODAG, FeatureExtractor
    from bioFunctionPrediction.src.utils.dataloader import DataIterator, DataLoader

log = logging.getLogger('predict')
FLAGS = tf.app.flags.FLAGS


def create_args():
    tf.app.flags.DEFINE_string(
        'resources',
        './resources',
        "path to data")

    tf.app.flags.DEFINE_string(
        'modelsdir',
        './output/savedmodels/',
        "path to model")

    tf.app.flags.DEFINE_string(
        'inputfile',
        'seqs.tar',
        "path to fasta sequence file")

    tf.app.flags.DEFINE_string(
        'function',
        'mf',
        'default function to run'
    )
    tf.app.flags.DEFINE_integer(
        'batchsize',
        128,
        'size of batch'
    )
    tf.app.flags.DEFINE_integer(
        'maxseqlen',
        2000,
        'maximum sequence length'
    )

    tf.app.flags.DEFINE_integer(
        'testsize',
        100,
        'number of validation batches to use'
    )
    tf.app.flags.DEFINE_boolean(
        'evaluate',
        False,
        'evaluate the results and output precision, recall'
    )
    return


def predict_evaluate(dataiter, thres, placeholders, modelpath):
    avgPrec, avgRecall, avgF1 = 0.0, 0.0, 0.0
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:

        saver = tf.train.import_meta_graph(
            glob(os.path.join(
                modelpath, 'model*meta'))[0])

        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored model')
        graph = tf.get_default_graph()
        # tf_x, tf_y, tf_thres = graph.get_tensor_by_name('x_input:0'), graph.get_tensor_by_name('y_out:0')
        # tf_thres = graph.get_tensor_by_name('thres:0')
        tf_x, tf_y, tf_thres = [graph.get_tensor_by_name(name) for name in placeholders]
        metrics = [graph.get_tensor_by_name('precision:0'),
                   graph.get_tensor_by_name('recall:0'),
                   graph.get_tensor_by_name('f1:0')]
        log.info('starting prediction')
        step = 0
        print ( 'Testing model...')

        # pdb.set_trace()
        for x, y in dataiter:
            if x.shape[0] != y.shape[0]:
                raise Exception('invalid, x-{}, y-{}'.format(str(x.shape), str(y.shape)))
            
            prec, recall, f1 = sess.run(metrics, feed_dict={tf_y: y, tf_x: x, tf_thres: thres})
            # pdb.set_trace()
            log.info('f1 for batch:{}'.format(f1))
            avgPrec += prec
            avgRecall += recall
            avgF1 += f1
            step += 1

        dataiter.close()
        # log.info('read {} test batches of size-{}'.format(step, x.shape[0]))
        if step > 0 :
            log.info('f1:{}'.format(avgF1 / step))
    try:
        avgPrec = avgPrec
        avgRecall = avgRecall / step 
        avgF1 = avgF1 / step
    except:
        log.error('Error in calcultion of metrics! possible divison by 0')

    return avgPrec , avgRecall, avgF1

def predict_validate(dataiter, thres, placeholders, modelpath):
    step = 0
    THRESHOLD_RANGE = np.arange(0.1, 0.5, 0.05)
    avgPrec, avgRecall, avgF1 = (np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE),
                                 np.zeros_like(THRESHOLD_RANGE)
                                 )
    new_graph = tf.Graph()
    if not isinstance(thres, list):
        thres = [thres]

    log.info('thres: {}'.format(thres))
    with tf.Session(graph=new_graph) as sess:
        saver = tf.train.import_meta_graph(glob(os.path.join(modelpath, 'model*meta'))[0])
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored validation model')
        graph = tf.get_default_graph()
        # tf_x, tf_y, tf_thres = graph.get_tensor_by_name('x_input:0'), graph.get_tensor_by_name('y_out:0')
        # tf_thres = graph.get_tensor_by_name('thres:0')
        tf_x, tf_y, tf_thres = [graph.get_tensor_by_name(name) for name in placeholders]
        metrics = [graph.get_tensor_by_name('precision:0'),
                   graph.get_tensor_by_name('recall:0'),
                   graph.get_tensor_by_name('f1:0')]
        log.info('starting prediction')
        step = 0
        for x, y in dataiter:
            prec, recall, f1 = [], [], []
            for thres in THRESHOLD_RANGE:
                p, r, f = sess.run(metrics, feed_dict={tf_y: y, tf_x: x, tf_thres: [thres]})
                # p, r, f, summary = sess.run([decoder.precision, decoder.recall,
                #                             decoder.f1score, decoder.summary],
                #                             feed_dict={decoder.ys_: y, encoder.xs_: x,
                #                                        decoder.threshold: [thres]})
                # summary_writer.add_summary(summary, step)
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


def print_predictions(predictions, gofuncs):
    for row in range(predictions.shape[0]):
        print(dict(zip(gofuncs.GOIDS, predictions[row, :])))

    return


def predict(dataiter, thres, modelpath, gofuncs):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(glob(os.path.join(modelpath, 'model*meta'))[0])
        saver.restore(sess, tf.train.latest_checkpoint(modelpath))
        log.info('restored model')
        graph = tf.get_default_graph()
        tf_x, tf_y = graph.get_tensor_by_name('x_in:0'), graph.get_tensor_by_name('y_out:0')
        tf_thres = graph.get_tensor_by_name('thres:0')
        prediction_prob = [graph.get_tensor_by_name('prediction:0')]
        log.info('starting prediction')
        step = 0

        for x, y in dataiter:
            predictions = sess.run(
                prediction_prob,
                feed_dict={
                    tf_y: y,
                    tf_x: x,
                    tf_thres: [thres]
                }
            )
            print_predictions(predictions, gofuncs)
            step += 1

        dataiter.close()
        log.info('read {} test batches'.format(step))
    return


def main(argv):
    log.info('Beginning prediction')
    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)

    log.info('GO DAG initialized. Updated function list-{}'.format(len(funcs)))
    FeatureExtractor.load(FLAGS.resources)
    log.info('Loaded amino acid and ngram mapping data')

    data = DataLoader(filename=FLAGS.inputfile)
    if FLAGS.evaluate:
        test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                     dataloader=data, functype=FLAGS.function, featuretype='ngrams')
        predict_evaluate(test_dataiter, 0.2, FLAGS.modelsdir)
    else:
        test_dataiter = DataIterator(batchsize=FLAGS.batchsize, size=FLAGS.testsize,
                                     dataloader=data, functype=FLAGS.function, featuretype='ngrams', test=True)
        predict(test_dataiter, 0.2, FLAGS.modelsdir, funcs)


if __name__ == "__main__":
    create_args()
    tf.app.run(main)
    # sample run command
    ## python predict.py --modelsdir ./savedmodels/deepGo --resources ./resources --testsize 100 -- batchsize 10
