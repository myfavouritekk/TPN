#!/usr/bin/env python

import argparse
import sys
import os
import os.path as osp
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/caffe-mpi/build/install/python'))
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
sys.path.insert(0, osp.join(this_dir, '../../src'))
from fast_rcnn.craft import im_detect
from fast_rcnn.config import cfg, cfg_from_file
import caffe
from caffe.proto import caffe_pb2
from mpi4py import MPI
import google.protobuf as protobuf
import yaml
import glob
from vdetlib.utils.protocol import proto_load, frame_path_at
from vdetlib.utils.common import imread
sys.path.insert(0, osp.join(this_dir, '../../src'))
from sequence_roi_data_layer.provider import SequenceROIDataProvider
import numpy as np
import cPickle
import random
import cv2

def parse_args():
    parser = argparse.ArgumentParser('TPN training.')
    parser.add_argument('model')
    parser.add_argument('weights')
    parser.add_argument('--iter', type=int)
    parser.add_argument('--val_cfg')
    parser.add_argument('--rcnn_cfg', default=None)
    parser.add_argument('--device_id', type=int, nargs='+', required=True)
    args = parser.parse_args()
    return args

def load_data(config_file):
    provider = SequenceROIDataProvider(config_file)
    return provider

def load_nets(args, cur_gpu):
    # initialize solver and feature net,
    # RNN should be initialized before CNN, because CNN cudnn conv layers
    # may assume using all available memory
    caffe.set_mode_gpu()
    caffe.set_device(cur_gpu)
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    return net

if __name__ == '__main__':
    args = parse_args()

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    pool_size = comm.Get_size()
    # TODO: wheather to check
    caffe.set_parallel()

    # load config file
    with open(args.val_cfg) as f:
        config = yaml.load(f.read())
    print "Config:\n{}".format(config)

    if args.rcnn_cfg is not None:
        cfg_from_file(args.rcnn_cfg)

    # load data
    provider = load_data(args.val_cfg)

    # get gpu id
    gpus = args.device_id
    assert len(gpus) >= pool_size
    cur_gpu = gpus[mpi_rank]
    cfg.GPU_ID = cur_gpu

    # load solver and nets
    net = load_nets(args, cur_gpu)

    # start training
    iter = mpi_rank
    provider.iter = iter
    losses = {}
    for i in xrange(args.iter):
        data, rois, labels, bbox_targets, bbox_weights = provider.forward(pool_size)

        net.blobs['data'].reshape(*(data.shape))
        net.blobs['data'].data[...] = data
        net.blobs['rois'].reshape(*(rois.shape))
        net.blobs['rois'].data[...] = rois
        net.blobs['labels'].reshape(*(labels.shape))
        net.blobs['labels'].data[...] = labels
        net.blobs['bbox_targets'].reshape(*bbox_targets.shape)
        net.blobs['bbox_targets'].data[...] = bbox_targets
        net.blobs['bbox_weights'].reshape(*bbox_weights.shape)
        net.blobs['bbox_weights'].data[...] = bbox_weights

        forward_out = net.forward()
        for loss_name in forward_out:
            loss = forward_out[loss_name]
            reduced_loss = np.array(0, 'float32')
            comm.Allreduce([loss, MPI.FLOAT], [reduced_loss, MPI.FLOAT], op=MPI.SUM)
            if loss_name not in losses:
                losses[loss_name] = 0
            losses[loss_name] += reduced_loss
    if mpi_rank == 0:
        for loss_name in losses:
            print "{}: {:.06f}".format(loss_name, losses[loss_name] / args.iter / pool_size)


