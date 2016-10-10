#!/usr/bin/env python

import sys
import os.path as osp
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/caffe-mpi/build/install/python/'))
import caffe
import argparse
import cPickle
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('def_file')
    parser.add_argument('param')
    parser.add_argument('bbox_mean')
    parser.add_argument('bbox_std')
    parser.add_argument('save_cls_param_file')
    parser.add_argument('save_bbox_param_file')
    args = parser.parse_args()

    net = caffe.Net(args.def_file, args.param, caffe.TEST)

    cls_w = net.params['cls_score_vid'][0].data.T
    cls_b = net.params['cls_score_vid'][1].data

    with open(args.save_cls_param_file, 'wb') as f:
        cPickle.dump((cls_w, cls_b), f)

    with open(args.bbox_mean, 'rb') as f:
        bbox_means = cPickle.load(f)
    with open(args.bbox_std, 'rb') as f:
        bbox_stds = cPickle.load(f)

    net.params['bbox_pred_vid'][0].data[...] = \
        net.params['bbox_pred_vid'][0].data * bbox_stds[:, np.newaxis]

    net.params['bbox_pred_vid'][1].data[...] = \
        net.params['bbox_pred_vid'][1].data * bbox_stds + bbox_means

    bbox_w = net.params['bbox_pred_vid'][0].data.T
    bbox_b = net.params['bbox_pred_vid'][1].data

    with open(args.save_bbox_param_file, 'wb') as f:
        cPickle.dump((bbox_w, bbox_b), f)
