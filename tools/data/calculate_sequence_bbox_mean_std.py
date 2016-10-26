#!/usr/bin/env python

import argparse
import scipy.io as sio
import sys
import os.path as osp
import numpy as np
import cPickle
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
from fast_rcnn.bbox_transform import bbox_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paired_gt_file')
    parser.add_argument('save_mean_file')
    parser.add_argument('save_std_file')
    args = parser.parse_args()

    deltas = []
    gts = sio.loadmat(args.paired_gt_file)['gt']
    for track_gts in gts:
        gt1 = track_gts[0]
        if len(gt1) == 0: continue
        cur_deltas = []
        for gt in track_gts[1:]:
            cur_deltas.append(bbox_transform(gt1, gt))
        deltas.append(np.hstack(cur_deltas))
    delta = np.vstack(deltas)
    mean = np.mean(delta, axis=0)
    std = np.std(delta, axis=0)
    with open(args.save_mean_file, 'wb') as f:
        cPickle.dump(mean, f, cPickle.HIGHEST_PROTOCOL)
    with open(args.save_std_file, 'wb') as f:
        cPickle.dump(std, f, cPickle.HIGHEST_PROTOCOL)
