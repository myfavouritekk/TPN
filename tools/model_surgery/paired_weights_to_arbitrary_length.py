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
    parser.add_argument('param_file')
    parser.add_argument('save_file')
    parser.add_argument('--box_layer',
        help='Box regression layer to change.')
    parser.add_argument('--cls_layer',
        help='Classification layer to change.')
    parser.add_argument('--length', type=int, default=3,
        help='New length. Should be greater than 2. [3]')
    args = parser.parse_args()

    net = caffe.Net(args.def_file, args.param_file, caffe.TEST)

    # bbox_regression layer
    weight = net.params[args.box_layer][0].data
    bias = net.params[args.box_layer][1].data

    length = args.length - 1

    # bias is direct repetition
    new_bias = np.tile(bias, length)
    net.params[args.box_layer][1].reshape(*new_bias.shape)
    net.params[args.box_layer][1].data[...] = new_bias

    # weight
    feat_dim = weight.shape[1] / 2
    frame1_weight = weight[:,:feat_dim]
    frame2_weight = weight[:,feat_dim:]
    new_weight = np.zeros((4 * length, feat_dim * (length + 1)))
    for i in xrange(length):
        new_weight[4*i:4*i+4,:feat_dim] = frame1_weight
        new_weight[4*i:4*i+4,feat_dim+i*feat_dim:2*feat_dim+i*feat_dim] = frame2_weight
    net.params[args.box_layer][0].reshape(*new_weight.shape)
    net.params[args.box_layer][0].data[...] = new_weight

    # classification layer
    cls_weight = net.params[args.cls_layer][0].data

    # cls_weight
    feat_dim = cls_weight.shape[1] / 2
    frame1_cls_weight = cls_weight[:,:feat_dim]
    frame2_cls_weight = cls_weight[:,feat_dim:]
    new_cls_weight = np.zeros((cls_weight.shape[0], feat_dim * (length + 1)))
    new_cls_weight[:,:feat_dim] = frame1_cls_weight
    for i in xrange(1,length+1):
        new_cls_weight[:,i*feat_dim:feat_dim+i*feat_dim] = frame2_cls_weight / float(length)
    net.params[args.cls_layer][0].reshape(*new_cls_weight.shape)
    net.params[args.cls_layer][0].data[...] = new_cls_weight

    net.save(args.save_file)

