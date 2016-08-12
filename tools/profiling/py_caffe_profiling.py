#!/usr/bin/env python

import sys
import os
import argparse
import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Time profiling of certain Caffe code.')
    parser.add_argument('--caffe',
        help='Path to caffe repository.')
    parser.add_argument('--gpu', type=int, default=0,
        help='GPU id. [0]')
    parser.add_argument('--model',
        help='Model prototxt.')
    parser.add_argument('--weights',
        help='Model parameter file (.caffemodel).')
    parser.add_argument('--iterations', type=int, default=50,
        help='Number of iterations. [50]')
    parser.add_argument('--size', type=int, default=700,
        help='Image size. [700]')
    parser.add_argument('--num_roi', type=int, default=128,
        help='Number of ROIs. [128]')
    args = parser.parse_args()

    # import caffe
    sys.path.insert(0, os.path.join(args.caffe, 'python'))
    print "Using caffe from {}".format(args.caffe)
    try:
        import caffe
    except ImportError:
        print "ImportError: {} seems not a caffe repository.".format(args.caffe)
        sys.exit()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)
    net = caffe.Net(args.model, args.weights, caffe.TEST)
    size = args.size
    num_roi = args.num_roi
    for i in xrange(args.iterations):
        st = time.time()
        net.blobs['data'].reshape(1, 3, size, size)
        net.blobs['rois'].reshape(num_roi, 5)
        net.forward()
        print "Iter {}: {:.02f} s for forward.".format(
            i+1, time.time() - st)

