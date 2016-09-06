#!/usr/bin/env python

import sys
import os.path as osp
code_root=osp.join(osp.dirname(__file__), '../..')
sys.path.insert(0, osp.join(code_root, 'src'))
sys.path.insert(0, osp.join(code_root, 'external/py-faster-rcnn/lib'))
sys.path.insert(0, osp.join(code_root, 'external'))
sys.path.insert(0, osp.join(code_root, 'external/caffe-mpi/build/install/python'))
import caffe
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.craft import im_detect
import argparse
from vdetlib.utils.protocol import proto_load, proto_dump, track_box_at_frame, frame_path_at
from vdetlib.utils.common import imread
from vdetlib.utils.visual import add_bbox
from tpn.propagate import tpn_caffe_test
import cPickle
import numpy as np
import cv2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='TPN End-to-end Testing.')
    parser.add_argument('vid_file')
    parser.add_argument('box_file')
    parser.add_argument('save_file')
    parser.add_argument('--job', dest='job_id', help='Job slot id. GPU id + 1. [1]',
                        default=1, type=int)
    parser.add_argument('--def', dest='def_file', help='Network definition file.')
    parser.add_argument('--param', help='Network parameter file.')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--lstm_def', help='LSTM definition file.')
    parser.add_argument('--lstm_param', help='LSTM parameter file.')
    parser.add_argument('--num_per_batch', dest='boxes_num_per_batch',
                        help='split boxes to batches',
                        default=64, type=int)
    parser.add_argument('--bbox_mean', dest='bbox_mean',
                        help='the    mean of bbox',
                        default=None, type=str)
    parser.add_argument('--bbox_std', dest='bbox_std',
                        help='the std of bbox',
                        default=None, type=str)
    parser.add_argument('--bbox_pred_layer', dest='bbox_pred_layer',
                        help='Layer name for bbox regression layer in feature net.',
                        default='bbox_pred_vid', type=str)
    parser.add_argument('--rnn_bbox_pred_layer', dest='rnn_pred_layer',
                        help='Layer name for bbox regression layer in rnn net.',
                        default='bbox_pred_vid', type=str)
    parser.add_argument('--scheme', help='Propagation scheme. [weighted]',
                        choices=['max', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--length', type=int, default=20,
                        help='Propagation length. [20]')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Temporal subsampling rate. [1]')
    parser.add_argument('--offset', type=int, default=0,
                        help='Sampling offset. [0]')
    parser.add_argument('--vis_debug', action='store_true')
    parser.add_argument('--gpus', nargs='+', default=None, type=int, help='Available GPUs.')
    parser.set_defaults(vis_debug=False)
    args = parser.parse_args()
    return args

def load_models(args):

    # load rnn model
    caffe.set_mode_gpu()
    if args.gpus is None:
        caffe.set_device(args.job_id - 1)
    else:
        assert args.job_id <= len(args.gpus)
        caffe.set_device(args.gpus[args.job_id-1])
    if args.lstm_param is not '':
        rnn_net = caffe.Net(args.lstm_def, args.lstm_param, caffe.TEST)
        print 'Loaded RNN network from {:s}.'.format(args.lstm_def)
    else:
        rnn_net = caffe.Net(args.lstm_def, caffe.TEST)
        print 'WARNING: dummy RNN network created.'

    # load feature model
    feature_net = caffe.Net(args.def_file, args.param, caffe.TEST)
    print 'Loaded feature network from {:s}.'.format(args.def_file)

    return feature_net, rnn_net

def show_tracks(vid_proto, track_proto):
    for frame in vid_proto['frames']:
        img = imread(frame_path_at(vid_proto, frame['frame']))
        boxes = [track_box_at_frame(tracklet, frame['frame']) \
                for tracklet in track_proto['tracks']]
        tracked = add_bbox(img, boxes, None, None, 2)
        cv2.imshow('tracks', tracked)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    if osp.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    cfg.DEDUP_BOXES = 0.0
    cfg.GPU_ID = args.job_id - 1

    # Load models
    feature_net, rnn_net = load_models(args)

    # Load protocols
    vid_proto = proto_load(args.vid_file)
    box_proto = proto_load(args.box_file)

    # apply bbox regression normalization on the net weights
    with open(args.bbox_mean, 'rb') as f:
        bbox_means = cPickle.load(f)
    with open(args.bbox_std, 'rb') as f:
        bbox_stds = cPickle.load(f)
    feature_net.params[args.bbox_pred_layer][0].data[...] = \
        feature_net.params[args.bbox_pred_layer][0].data * bbox_stds[:, np.newaxis]
    feature_net.params[args.bbox_pred_layer][1].data[...] = \
        feature_net.params[args.bbox_pred_layer][1].data * bbox_stds + bbox_means
    rnn_net.params[args.rnn_pred_layer][0].data[...] = \
        rnn_net.params[args.rnn_pred_layer][0].data * bbox_stds[:, np.newaxis]
    rnn_net.params[args.rnn_pred_layer][1].data[...] = \
        rnn_net.params[args.rnn_pred_layer][1].data * bbox_stds + bbox_means

    # End-to-end testing
    track_proto = tpn_caffe_test(vid_proto, box_proto, feature_net, rnn_net,
        im_detect, scheme=args.scheme,
        length=args.length, sample_rate=args.sample_rate,
        offset=args.offset, batch_size=args.boxes_num_per_batch)

    if args.vis_debug:
        show_tracks(vid_proto, track_proto)

    proto_dump(track_proto, args.save_file)
