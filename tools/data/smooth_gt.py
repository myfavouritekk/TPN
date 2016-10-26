#!/usr/bin/env python

import sys
import os.path as osp
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/'))
from vdetlib.utils.protocol import proto_load, proto_dump
import argparse
import numpy as np
import scipy.ndimage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_file')
    parser.add_argument('save_file')
    parser.add_argument('--window', type=int, default=11)
    args = parser.parse_args()

    annot_proto = proto_load(args.gt_file)
    for annot in annot_proto['annotations']:
        boxes = np.asarray([box['bbox'] for box in annot['track']], dtype=np.float)
        smoothed = scipy.ndimage.filters.gaussian_filter1d(boxes, args.window / 6.,
            axis=0, mode='nearest')
        for box, pred_bbox in zip(annot['track'], smoothed):
            box['bbox'] = pred_bbox.tolist()

    proto_dump(annot_proto, args.save_file)