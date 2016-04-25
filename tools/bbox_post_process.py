#!/usr/bin/env python
import init
import caffe
import numpy as np
import argparse
import cPickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net_def')
    parser.add_argument('net_param')
    parser.add_argument('save_file')
    parser.add_argument('--bbox_means', default='bbox_means.pkl')
    parser.add_argument('--bbox_stds', default='bbox_stds.pkl')
    args = parser.parse_args()

    net = caffe.Net(args.net_def, args.net_param, caffe.TEST)
    with open(args.bbox_means, 'rb') as f:
        bbox_means = cPickle.load(f)
    with open(args.bbox_stds, 'rb') as f:
        bbox_stds = cPickle.load(f)
    net.params['bbox_pred'][0].data[...] = net.params['bbox_pred'][0].data * bbox_stds[:, np.newaxis]
    net.params['bbox_pred'][1].data[...] = net.params['bbox_pred'][1].data * bbox_stds + bbox_means

    print "Saved to {}.".format(args.save_file)
    net.save(args.save_file)