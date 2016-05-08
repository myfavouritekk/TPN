#!/usr/bin/env python

import os, sys, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import init
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import cPickle

from vdetlib.utils.protocol import proto_load, proto_dump, frame_path_at
from vdetlib.vdet.dataset import index_vdet_to_det
from tpn.propagate import roi_propagation

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Regression propagation.')
    parser.add_argument('vid_file')
    parser.add_argument('box_file')
    parser.add_argument('save_file')
    parser.add_argument('--job', dest='job_id', help='Job slot id. GPU id + 1. [1]',
                        default=1, type=int)
    parser.add_argument('--def', dest='def_file', help='Network defination file.')
    parser.add_argument('--param', help='Network parameter file.')
    parser.add_argument('--scheme', help='Propagation scheme. [weighted]',
                        choices=['max', 'mean', 'weighted'], default='weighted')
    parser.add_argument('--length', type=int, default=9,
                        help='Propagation length. [9]')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Temporal subsampling rate. [1]')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # cfg.TEST.HAS_RPN = False  # Use RPN for proposals
    cfg.DEDUP_BOXES = 0

    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.job_id - 1)

    net = caffe.Net(args.def_file, args.param, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(args.def_file)

    vid_proto = proto_load(args.vid_file)
    box_proto = proto_load(args.box_file)

    track_proto = roi_propagation(vid_proto, box_proto, net, scheme=args.scheme,
        length=args.length, sample_rate=args.sample_rate, cls_indices=index_vdet_to_det.values(),
        keep_feat=True)

    proto_dump(track_proto, args.save_file)
