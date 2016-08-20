#!/usr/bin/env python

# --------------------------------------------------------
# Test regression propagation on ImageNet VID video
# Modified by Kai KANG (myfavouritekk@gmail.com)
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import argparse
import pprint
import time
import os
import os.path as osp
import sys
import cPickle
import numpy as np
import scipy.io as sio

this_dir = osp.dirname(__file__)
# add caffe-mpi path
sys.path.insert(0, osp.join(this_dir, '../../external/caffe-mpi/build/install/python'))
import caffe

# add py-faster-rcnn paths
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
from fast_rcnn.craft import im_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb

# add external libs
sys.path.insert(0, osp.join(this_dir, '../../external'))
from vdetlib.utils.protocol import proto_load
from vdetlib.vdet.dataset import imagenet_vdet_classes
from vdetlib.vdet.video_det import fast_rcnn_det_vid_raw

# add src libs
sys.path.insert(0, osp.join(this_dir, '../../src'))
from tpn.propagate import roi_propagation
from tpn.target import add_track_targets
from tpn.data_io import save_track_proto_to_zip

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('vid_file')
    parser.add_argument('box_file')
    parser.add_argument('save_file', help='Save zip file')
    parser.add_argument('--annot_file', default=None,
                        help='Ground truth annotation file. [None]')
    parser.add_argument('--job', dest='job_id', help='Job slot, GPU ID + 1. [1]',
                        default=1, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--num_per_batch', dest='boxes_num_per_batch',
                        help='split boxes to batches. [32]',
                        default=32, type=int)
    parser.add_argument('--bbox_mean', dest='bbox_mean',
                        help='the mean of bbox',
                        default=None, type=str)
    parser.add_argument('--bbox_std', dest='bbox_std',
                        help='the std of bbox',
                        default=None, type=str)
    parser.add_argument('--scheme', help='Propagation scheme. [weighted]',
                        choices=['max', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--length', type=int, default=9,
                        help='Propagation length. [9]')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Temporal subsampling rate. [1]')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset of sampling. [0]')
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.set_defaults(vis=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.job_id - 1

    print 'Using config:'
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print 'Waiting for {} to exist...'.format(args.caffemodel)
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.job_id - 1)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    # apply bbox regression normalization on the net weights
    with open(args.bbox_mean, 'rb') as f:
        bbox_means = cPickle.load(f)
    with open(args.bbox_std, 'rb') as f:
        bbox_stds = cPickle.load(f)

    net.params['bbox_pred_vid'][0].data[...] = \
        net.params['bbox_pred_vid'][0].data * bbox_stds[:, np.newaxis]

    net.params['bbox_pred_vid'][1].data[...] = \
        net.params['bbox_pred_vid'][1].data * bbox_stds + bbox_means

    vid_proto = proto_load(args.vid_file)
    box_proto = proto_load(args.box_file)

    track_proto = roi_propagation(vid_proto, box_proto, net, im_detect, scheme=args.scheme,
        length=args.length, sample_rate=args.sample_rate,
        keep_feat=True, batch_size=args.boxes_num_per_batch)

    # add ground truth targets if annotation file is given
    if args.annot_file is not None:
        annot_proto = proto_load(args.annot_file)
        add_track_targets(track_proto, annot_proto)

    save_track_proto_to_zip(track_proto, args.save_file)
