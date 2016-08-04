#!/usr/bin/env python

import sys
sys.path.insert(0, '/Volumes/Research/ImageNet2016/Code/external/kwang/py-faster-rcnn-craft/caffe-fast-rcnn/python/')
sys.path.insert(0, 'src')
sys.path.insert(0, 'external/py-faster-rcnn/lib')
sys.path.insert(0, 'external')
import caffe
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
import argparse
from tpn.recurrent_extract_features import TestConfig
from tpn.model import TPNModel
from tpn.propagate import tpn_test
from vdetlib.utils.protocol import proto_load, proto_dump

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='TPN End-to-end Testing.')
    parser.add_argument('vid_file')
    parser.add_argument('box_file')
    parser.add_argument('save_file')
    parser.add_argument('--job', dest='job_id', help='Job slot id. GPU id + 1. [1]',
                        default=1, type=int)
    parser.add_argument('--def', dest='def_file', help='Network defination file.')
    parser.add_argument('--param', help='Network parameter file.')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--lstm_path', help='Path to LSTM model')
    parser.add_argument('--lstm_type', type=str, choices=['residual', 'basic'], default='basic',
        help='Type of LSTM cells. [basic]')
    parser.add_argument('--lstm_num', type=int, default=1,
        help='Number of LSTM layer(s). [1]')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--num_per_batch', dest='boxes_num_per_batch',
                        help='split boxes to batches',
                        default=0, type=int)
    parser.add_argument('--bbox_mean', dest='bbox_mean',
                        help='the mean of bbox',
                        default=None, type=str)
    parser.add_argument('--bbox_std', dest='bbox_std',
                        help='the std of bbox',
                        default=None, type=str)
    parser.add_argument('--scheme', help='Propagation scheme. [weighted]',
                        choices=['max', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--length', type=int, default=20,
                        help='Propagation length. [20]')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Temporal subsampling rate. [1]')
    parser.add_argument('--offset', type=int, default=0,
                        help='Sampling offset. [0]')
    args = parser.parse_args()
    return args

def load_models(args):

    # load rnn model
    config = TestConfig()
    config.num_layers = args.lstm_num
    config.type = args.lstm_type

    #tf.set_random_seed(1017)
    sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth=True
    with tf.Graph().as_default():
        session = tf.Session(config=sess_config)
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=None):
            # with tf.device("/gpu:{}".format(args.job_id)):
            with tf.device("/cpu"):
                rnn_net = TPNModel(is_training=False, config=config)

        # restoring variables
        saver = tf.train.Saver()
        print 'Loaded RNN network from {:s}.'.format(args.lstm_path)
        saver.restore(session, args.lstm_path)

    # load feature model
    caffe.set_mode_gpu()
    caffe.set_device(args.job_id - 1)
    feature_net = caffe.Net(args.def_file, args.param, caffe.TEST)
    print 'Loaded feature network from {:s}.'.format(args.def_file)

    return feature_net, rnn_net, session

if __name__ == '__main__':
    args = parse_args()

    # Load models
    cfg.DEDUP_BOXES = 0
    feature_net, rnn_net, session = load_models(args)

    # Load protocols
    vid_proto = proto_load(args.vid_file)
    box_proto = proto_load(args.box_file)

    # End-to-end testing
    track_proto = tpn_test(vid_proto, box_proto, feature_net, rnn_net,
        session, im_detect, scheme=args.scheme,
        length=args.length, sample_rate=args.sample_rate,
        offset=args.offset)

    proto_dump(track_proto, args.save_file)
