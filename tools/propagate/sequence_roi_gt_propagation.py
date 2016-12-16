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

this_dir = osp.dirname(__file__)

# add py-faster-rcnn paths
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
from fast_rcnn.craft import sequence_im_detect
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list

# add external libs
sys.path.insert(0, osp.join(this_dir, '../../external'))
from vdetlib.utils.protocol import proto_load, proto_dump

# add src libs
sys.path.insert(0, osp.join(this_dir, '../../src'))
from tpn.propagate import gt_motion_propagation
from tpn.target import add_track_targets
from tpn.data_io import save_track_proto_to_zip

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('vid_file')
    parser.add_argument('box_file')
    parser.add_argument('annot_file', default=None,
                        help='Ground truth annotation file. [None]')
    parser.add_argument('save_file', help='Save zip file')
    parser.add_argument('--job', dest='job_id', help='Job slot, GPU ID + 1. [1]',
                        default=1, type=int)
    parser.add_argument('--length', type=int, default=20,
                        help='Propagation length. [20]')
    parser.add_argument('--window', type=int, default=5,
                        help='Prediction window. [5]')
    parser.add_argument('--sample_rate', type=int, default=1,
                        help='Temporal subsampling rate. [1]')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset of sampling. [0]')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='GT overlap threshold for tracking. [0.5]')
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.set_defaults(vis=False, zip=False, keep_feat=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    if osp.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(1)

    vid_proto = proto_load(args.vid_file)
    box_proto = proto_load(args.box_file)
    annot_proto = proto_load(args.annot_file)

    track_proto = gt_motion_propagation(vid_proto, box_proto, annot_proto,
        window=args.window, length=args.length,
        sample_rate=args.sample_rate, overlap_thres=args.overlap)

    # add ground truth targets if annotation file is given
    add_track_targets(track_proto, annot_proto)

    if args.zip:
        save_track_proto_to_zip(track_proto, args.save_file)
    else:
        proto_dump(track_proto, args.save_file)
