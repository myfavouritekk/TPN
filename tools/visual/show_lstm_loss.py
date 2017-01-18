#!/usr/bin/env python

import sys, os
import argparse
from vdetlib.utils.protocol import proto_load
import tensorflow as tf
import numpy as np
sys.path.insert(0, 'src')
from tpn.data_io import tpn_test_iterator
from tpn.model import EncoderDecoderModel as TPNModel
sys.path.insert(0, 'external/py-faster-rcnn/lib')
from fast_rcnn.bbox_transform import bbox_transform_inv
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class TestConfig(object):
    """Default config."""
    init_scale = 0.01
    learning_rate = 0.001
    momentum = 0.9
    max_grad_norm = 1.5
    num_steps = 20
    input_size = 1024
    hidden_size = 1024
    max_epoch = 5
    iter_epoch = 2000
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    num_classes = 31
    cls_weight = 1.0
    bbox_weight = 0.0
    ending_weight = 1.0
    vid_per_batch = 4
    cls_init = ''
    bbox_init = ''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('track_file')
    parser.add_argument('annot_file')
    parser.add_argument('lstm_model')
    parser.add_argument('num_layers', type=int,
        help='Number of layers')
    parser.add_argument('input_size', type=int,
        help='Input size.')
    parser.add_argument('save_fig',
        help='Save figure.')
    parser.add_argument('--type', type=str, choices=['residual', 'basic'], default='residual',
        help='Type of LSTM cells. [residual]')

    args = parser.parse_args()

    tracks = tpn_test_iterator(args.track_file)
    annot_proto = proto_load(args.annot_file)

    config = TestConfig()
    config.num_layers = args.num_layers
    config.type = args.type
    config.hidden_size = config.input_size = args.input_size
    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(
            -config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=None):
            m = TPNModel(is_training=False, config = config)

        saver = tf.train.Saver()
        saver.restore(session, args.lstm_model)

        cls_losses = []
        cls_tot_labels = []
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        y_trends = []
        z_trends = []
        for ind, track in enumerate(tracks, start=1):
            cls_labels = track['class_label']
            if np.any(cls_labels == -1):
                continue
            if np.all(np.logical_or(cls_labels == 0, cls_labels == -1)):
                continue
            # process track data
            track_length = track['feature'].shape[0]
            expend_feat = np.zeros((m.num_steps,) + track['feature'].shape[1:])
            expend_feat[:track_length] = track['feature']
        
            # extract features
            state = session.run([m.initial_state])
            cls_scores, bbox_deltas, end_probs, state = session.run(
            [m.cls_scores, m.bbox_pred, m.end_probs, m.final_state],
                {m.input_data: expend_feat[np.newaxis,:,:],
                m.initial_state: state[0]})
        
            # process outputs
            gt_len = cls_labels.shape[0]
            T = np.arange(gt_len)
            # bbox_pred = bbox_transform_inv(track['roi'], bbox_deltas[:gt_len,:])
            # cls_pred_lstm = np.argmax(cls_scores, axis=1)[:gt_len]
            # end_probs = end_probs[:gt_len]
        
            # calculate accuracy comparison
            # class labels may be -1
            cls_pred_loss = -np.log(cls_scores[T, cls_labels])
            # loss of ignored frames set to 0
            cls_pred_loss[cls_labels == -1] = 0
            cls_losses.append(cls_pred_loss.copy())
            cls_tot_labels.append(cls_labels.copy())
            # plot 1-order trend
            y = np.poly1d(np.polyfit(T, cls_pred_loss, 1))
            z = np.poly1d(np.polyfit(T, cls_labels!=0, 1))
            plt.plot(T, y(T))
            # ax.plot(T, y(T), z(T))
            # y_trends.append(y.c[0])
            # z_trends.append(z.c[0])

        # plt.scatter(y_trends, z_trends)
        # plt.plot(y_trends, z_trends, "o")
        # plt.show()
        plt.savefig(args.save_fig)


