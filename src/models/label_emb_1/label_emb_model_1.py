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
import sys
import pdb

sys.path.append('./..')
sys.path.append('./../..')

try:
    from utils.dataloader import GODAG, FeatureExtractor
    from utils.dataloader import DataIterator, DataLoader
    from models.conv_autoencoder import ConvAutoEncoder
    from models.conv_autoencoder_v1_0 import ConvAutoEncoder
    from models.label_emb_1.joint_inf_decoder import joint_inf_decoder
except:
    from bioFunctionPrediction.src.utils.dataloader import GODAG, FeatureExtractor
    from bioFunctionPrediction.src.utils.dataloader import DataIterator, DataLoader
    from bioFunctionPrediction.src.models.conv_autoencoder_v1_0 import ConvAutoEncoder
    from bioFunctionPrediction.src.models.label_emb_1.joint_inf_decoder import joint_inf_decoder
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('root')
FLAGS = tf.app.flags.FLAGS
THRESHOLD_RANGE = np.arange(0.10, 0.40, 0.05)
target_funcs_file_prefix = 'target_functions'


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
    tf.app.flags.DEFINE_integer(
        'predict',
        0,
        'whether to just check the model with test data'
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
        onlyLeafNodes=True
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
def validate(
        dataiter, sess, x_inp, decoder,
        summary_writer, target_funcs, label_emb
):
    global THRESHOLD_RANGE
    log.info(' Inside validate ::')
    step = 0
    avgPrec, avgRecall, avgF1 = (
        np.zeros_like(THRESHOLD_RANGE),
        np.zeros_like(THRESHOLD_RANGE),
        np.zeros_like(THRESHOLD_RANGE)
    )

    for x, y in dataiter:
        y_emb = convert_y_to_emb(y, target_funcs, label_emb)
        y_1hot = convert_y_to_1hot(y, target_funcs)

        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f, summary = sess.run(
                [decoder.b_prec,
                 decoder.b_recall,
                 decoder.b_f1,
                 decoder.summary
                 ],
                feed_dict={
                    x_inp: x,
                    decoder.y_labels: y_1hot,
                    decoder.y_inp: y_emb,
                    decoder.cos_sim_threshold: thres
                })
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


def test(dataiter, sess, x_inp, decoder,
         summary_writer, target_funcs, label_emb):
    global THRESHOLD_RANGE
    step = 0
    avgPrec, avgRecall, avgF1 = (
        np.zeros_like(THRESHOLD_RANGE),
        np.zeros_like(THRESHOLD_RANGE),
        np.zeros_like(THRESHOLD_RANGE)
    )

    for x, y in dataiter:
        y_1hot = convert_y_to_1hot(y, target_funcs)
        y_emb = convert_y_to_emb(y, target_funcs, label_emb)
        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f, summary = sess.run(
                [decoder.b_prec,
                 decoder.b_recall,
                 decoder.b_f1,
                 decoder.summary
                 ],
                feed_dict={
                    x_inp: x,
                    decoder.y_labels: y_1hot,
                    decoder.y_inp: y_emb,
                    decoder.cos_sim_threshold: thres
                })
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


# ------------------------------------ #

def get_word2vec_emb():
    global w2v_emb_dim
    file_loc = './../word2vec_1'
    file_name = 'GO_word_embed_dict.pkl'
    with open(os.path.join(file_loc, file_name), 'rb') as file_handle:
        emb_dict = pickle.load(file_handle)
        w2v_emb_dim = len(emb_dict[list(emb_dict.keys())[0]])
        return emb_dict


# ------------------------------------ #

def create_label_emb_lookup():
    global w2v_emb_dim
    w2v_emb_dict = get_word2vec_emb()

    # format the keys to add 'GO:'
    w2v_emb_dict = {
        'GO:' + k: v for k, v in w2v_emb_dict.items()
    }

    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    target_funcs = funcs
    FeatureExtractor.load(FLAGS.resources)

    label_count = len(target_funcs)
    _idmap = GODAG.idmap
    yid_vec_lookup = np.zeros([label_count + 1, w2v_emb_dim])

    for i in range(label_count):
        idx = _idmap[target_funcs[i]]
        try:
            yid_vec_lookup[idx] = w2v_emb_dict[target_funcs[i]]
        except:
            pass
    return yid_vec_lookup


# ------------------------------------ #

def filter_y(y, target_funcs):
    y = y * (~(y > len(target_funcs)))
    return y


# ------------------------------------ #

def dump_resources(outputdir):
    # dump the amino acid mapping
    FeatureExtractor.dump(outputdir)

    # dump  go id mapping
    GODAG.dump(outputdir)
    return


# ------------------------------------ #

# Convert y from [batch * list_of_ids]
# To : [batch * num_labels * emb_dimension]
def convert_y_to_emb(y, target_funcs, label_emb):
    global w2v_emb_dim
    y = filter_y(y, target_funcs)

    res_y = np.zeros([
        y.shape[0],
        label_emb.shape[0],
        label_emb.shape[1]
    ])

    for i in range(y.shape[0]):
        _y = y[i]
        mask = np.zeros(label_emb.shape)
        mask[_y] = 1.0
        _y = label_emb * mask
        res_y[i] = _y
    return res_y


# ------------------------------------ #

def convert_y_to_1hot(y, target_funcs):
    y = filter_y(y, target_funcs)
    res_y = np.zeros([y.shape[0], len(target_funcs) + 1])
    for i in range(y.shape[0]):
        _y = y[i]
        res_y[i, _y] = 1.0
    return res_y


# ------------------------------------ #

def build_model():
    global w2v_emb_dim
    global target_funcs_file_prefix
    
    target_funcs_file = '{}_{}.pkl'.format(target_funcs_file_prefix,FLAGS.function)

    funcs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format((FLAGS.function).lower())))['functions'].values
    target_funcs = funcs
    funcs = GODAG.initialize_idmap(funcs, FLAGS.function)
    FeatureExtractor.load(FLAGS.resources)

    with open(os.path.join(FLAGS.outputdir, target_funcs_file), 'wb') as fhandle:
        pickle.dump(target_funcs, fhandle)

    train_iter, test_iter, valid_iter = get_iterators()
    sess, cae_model_obj, x_inp, encoder_op = get_pretrained_cae()

    label_emb = create_label_emb_lookup()
    num_labels = len(target_funcs) + 1
    x_shape = encoder_op.shape.as_list()[1:]

    y_shape = [
        num_labels,
        w2v_emb_dim
    ]

    decoder = joint_inf_decoder(
        encoder_op,
        x_shape,
        y_shape,
        num_labels,
        w2v_emb_dim
    )
    decoder.build()

    init = tf.global_variables_initializer()
    init.run(session=sess)
    init_l = tf.local_variables_initializer()
    init_l.run(session=sess)
    chkpt = tf.train.Saver(max_to_keep=4)
    train_writer = tf.summary.FileWriter(
        FLAGS.outputdir + '/train', sess.graph
    )
    test_writer = tf.summary.FileWriter(FLAGS.outputdir + '/test')

    # ------------- Train --------------- #
    step = 0
    validation_check_steps = 800
    maxwait = 10
    best_f1 = 0.0
    wait = 0
    best_thres = 0
    metagraphFlag = True

    model_savename = 'label_emb_1_{}_{}'.format(int(time.time()),FLAGS.function)
    meta_graph_file_name = os.path.join(
        FLAGS.outputdir,
        model_savename,
        'model_{}.meta'.format(FLAGS.function)
    )
    tf.train.export_meta_graph(filename=meta_graph_file_name)

    # Best threshold at the moment
    last_threshold = None

    log.info('starting epochs')
    for epoch in range(FLAGS.num_epochs):
        for x, y in train_iter:
            if x.shape[0] != y.shape[0]:
                raise Exception(
                    'invalid, x :: {} , y :: {}'.format(str(x.shape), str(y.shape))
                )

            y_emb = convert_y_to_emb(y, target_funcs, label_emb)
            y_1hot = convert_y_to_1hot(y, target_funcs)

            _, total_loss = sess.run(
                [decoder.train, decoder.batch_loss],
                feed_dict={
                    x_inp: x,
                    decoder.y_inp: y_emb
                }
            )

            log.info('step :: {}, total_loss :: {} '.format(step, np.round(total_loss, 3)))
            step += 1

            if step % (validation_check_steps) == 0:
                log.info('beginning validation')
                prec, recall, f1 = validate(
                    valid_iter,
                    sess,
                    x_inp,
                    decoder,
                    test_writer,
                    target_funcs,
                    label_emb
                )
                thres = np.argmax(np.round(f1, 3))
                log.info('epoch: {} \n precision: {}, recall: {}, f1: {}'.format(
                    epoch,
                    np.round(prec, 2)[thres],
                    np.round(recall, 2)[thres],
                    np.round(f1, 2)[thres])
                )
                log.info('Threshold selected is {} '.format(THRESHOLD_RANGE[thres]))
                last_threshold = THRESHOLD_RANGE[thres]

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

                else:
                    wait += 1
                    if wait > maxwait:
                        log.info('f1 didnt improve for last {} validation steps, so stopping'.format(maxwait))
                        break

        train_iter.reset()

    outputdir = FLAGS.outputdir
    dump_resources(outputdir)

    log.info('----:: Testing Model ::----')
    prec, recall, f1 = test(
        test_iter,
        sess,
        x_inp,
        decoder,
        test_writer,
        target_funcs,
        label_emb
    )
    print('----------------')
    log.info('Test results')
    log.info('Threshold : {} '.format( THRESHOLD_RANGE ))
    log.info('Precision : {} '.format( np.round(prec, 4) ))
    log.info('Recall    : {} '.format( np.round(recall, 4) ))
    log.info('F1        : {} '.format( np.round(f1, 4) ))
    print('----------------')
    return


# ------------------------------------------------- #

# Returns the directory of complete pre-trained model and the name of the meta file
def get_pretrained_dir(model_name_prefix):
    z = glob.glob(os.path.join(FLAGS.outputdir, model_name_prefix + '*'))

    if len(z) > 0:
        print('Complete Pretrained model exists!')
        log.info('Complete Pretrained model exists!')
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


# This function is to test the final results only
def predict_only():
    log.info('Running prediction')
    global w2v_emb_dim
    global target_funcs_file_prefix

    # --- Load in saved data --- #
    target_funcs_file = '{}_{}.pkl'.format(target_funcs_file_prefix, FLAGS.function)

    with open(os.path.join(FLAGS.outputdir, 'GO_IDMAPPING.json')) as inf:
        idmapping = json.load(inf)
    GODAG.initialize_idmap(None, None, idmapping=idmapping)
    # read in target functions
    with open(os.path.join(FLAGS.outputdir, target_funcs_file), 'rb') as fhandle:
        target_funcs = pickle.load(fhandle)

    # load ngram mapping
    FeatureExtractor.load(FLAGS.outputdir)

    if FLAGS.featuretype == 'onehot':
        vocabsize = len(FeatureExtractor.aminoacidmap)
    else:
        vocabsize = len(FeatureExtractor.ngrammap)

    # -------------------------- #
    best_threshold = 0.2
    label_emb = create_label_emb_lookup()
    num_labels = len(target_funcs) + 1
    _, test_iter, _ = get_iterators()

    model_name_prefix = 'label_emb_1_cae'
    cae_pretrained_dir, cae_meta_file = get_cae_pretrained_dir(model_name_prefix)

    if cae_pretrained_dir is None:
        log.error( 'No Pretrained model !!! What are you doing dude ?!')
        exit(0)

    cae_model_obj = ConvAutoEncoder(
        vocab_size=vocabsize,
        maxlen=test_iter.expectedshape,
        batch_size=FLAGS.batchsize,
        embedding_dim=512
    )
    cae_model_obj.build(cae_pretrained_dir)

    model_name_prefix = 'label_emb_1_{}'.format(FLAGS.function)
    complete_pretrained_dir, complete_meta_file = get_pretrained_dir(model_name_prefix)

    x_inp = None
    encoder_op = None
    y_inf_inp = None

    tf.reset_default_graph()
    sess = tf.Session()

    if complete_pretrained_dir is None :
        log.error( 'No Pretrained model !!! What are you doing dude ?!')
        exit(0)

    saver = tf.train.import_meta_graph(complete_meta_file)
    saver.restore(sess, tf.train.latest_checkpoint(complete_pretrained_dir))
    graph = tf.get_default_graph()
    cae_names = cae_model_obj.return_wt_list_to_restore()

    for n in cae_names[1:]:
        try:
            graph.get_tensor_by_name(n)
            print(n)
        except:
            print(' Error restoring ..', n)

    x_inp = graph.get_tensor_by_name("model_input/x:0")
    encoder_op = graph.get_tensor_by_name("Encoder/Relu_4:0")

    x_shape = encoder_op.shape.as_list()[1:]

    y_shape = [
        num_labels,
        w2v_emb_dim
    ]

    decoder = joint_inf_decoder(
        encoder_op,
        x_shape,
        y_shape,
        num_labels,
        w2v_emb_dim
    )
    decoder.build(complete_pretrained_dir)
    jinf_names = decoder.return_wt_list_to_restore()
    print('-----')

    for n in jinf_names[1:]:
        if n in cae_names:
            continue
        try:
            graph.get_tensor_by_name(n)
        except:
            print(' Error restoring ..', n)

    y_inf_inp = graph.get_tensor_by_name("y_inf_inp:0")
    y_labels = graph.get_tensor_by_name("y_labels:0")
    threshold = graph.get_tensor_by_name("cos_sim_threshold:0")

    init = tf.global_variables_initializer()
    init.run(session=sess)
    init_l = tf.local_variables_initializer()
    init_l.run(session=sess)

    # ----------- #
    # Predict (Serenity Now!!)
    # ----------- #
    step = 0
    avg_prec = 0.0
    avg_recall =0.0
    avg_f1 = 0.0
    for x, y in test_iter:
        y_emb = convert_y_to_emb(y, target_funcs, label_emb)
        y_1hot = convert_y_to_1hot(y, target_funcs)

        _prec, _recall , _f1 = sess.run([
            decoder.b_prec,
            decoder.b_recall,
            decoder.b_f1
        ],feed_dict={
                x_inp: x,
                y_inf_inp : y_emb,
                y_labels: y_1hot,
                threshold: best_threshold
        })
        log.info(' Precision {} , Recall {} , F1 {} ' .format(
            np.round(_prec,4),
            np.round(_recall,4) ,
            np.round(_f1,4)
        ))
        log.info( ' ----- ')
        avg_f1 += _f1
        avg_prec += _prec
        avg_recall += _recall
        step += 1

    log.info(' ------------- ')
    log.info('Test set  Precision {} , Recall {} , F1 {} '.format(
        np.round(avg_prec/step, 4),
        np.round(avg_recall/step, 4),
        np.round(avg_f1/step, 4)
    ))

    return


# ------------------------------------------------- #

def main(argv):
    if FLAGS.predict == 0:
        build_model()
    else:
        predict_only()


if __name__ == "__main__":
    create_args()
    tf.app.run(main)
