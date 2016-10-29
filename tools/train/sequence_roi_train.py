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
from vdetlib.utils.visual import add_bbox
sys.path.insert(0, osp.join(this_dir, '../../src'))
from sequence_roi_data_layer.provider import SequenceROIDataProvider
import numpy as np
import cPickle
import random
import cv2

def parse_args():
    parser = argparse.ArgumentParser('TPN training.')
    parser.add_argument('solver')
    parser.add_argument('--train_cfg')
    parser.add_argument('--rcnn_cfg', default=None)
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument('--weights', type=str, default=None,
        help='RNN trained models.')
    restore.add_argument('--snapshot', type=str, default=None,
        help='RNN solverstates.')
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
    solver = caffe.SGDSolver(args.solver)
    if args.snapshot:
        print "Restoring history from {}".format(args.snapshot)
        solver.restore(args.snapshot)
    net = solver.net
    if args.weights:
        print "Copying weights from {}".format(args.weights)
        net.copy_from(args.weights)

    return solver, net

if __name__ == '__main__':
    args = parse_args()

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    pool_size = comm.Get_size()
    caffe.set_parallel()

    # load config file
    with open(args.train_cfg) as f:
        config = yaml.load(f.read())
    print "Config:\n{}".format(config)

    if args.rcnn_cfg is not None:
        cfg_from_file(args.rcnn_cfg)

    # load data
    provider = load_data(args.train_cfg)

    # read solver file
    solver_param = caffe_pb2.SolverParameter()
    with open(args.solver, 'r') as f:
        protobuf.text_format.Merge(f.read(), solver_param)
    max_iter = solver_param.max_iter

    # get gpu id
    gpus = solver_param.device_id
    assert len(gpus) >= pool_size
    cur_gpu = gpus[mpi_rank]
    cfg.GPU_ID = cur_gpu

    # load solver and nets
    solver, net = load_nets(args, cur_gpu)

    # start training
    iter = mpi_rank
    st_iter = solver.iter
    for i in xrange(max_iter - st_iter):
        data, rois, labels, bbox_targets, bbox_weights = provider.forward()

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

        solver.step(1)
