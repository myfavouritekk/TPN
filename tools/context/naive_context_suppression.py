#!/usr/bin/env python

import argparse
import numpy as np
from tpn.data_io import tpn_test_iterator
import os
import cPickle

def parse_args():
    parser = argparse.ArgumentParser('Naive context suppression: add bonus scores to the top classes.')
    parser.add_argument('input_track')
    parser.add_argument('output_track')
    parser.add_argument('--top_ratio', type=float, default=0.0003,
        help='Ratio of top detection. [0.0003]')
    parser.add_argument('--top_bonus', type=float, default=0.4,
        help='Bonus score for top classes. [0.4]')
    parser.add_argument('--score_key', type=str,
        help='Key name for detection scores.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    tracks = tpn_test_iterator(args.input_track)
    all_scores = np.concatenate([track[args.score_key] for track in tracks])
    num_box, num_cls = all_scores.shape
    all_cls_idx = np.tile(np.arange(num_cls), (num_box,1))

    # remove __background__
    all_scores = all_scores[:,1:].flatten()
    all_cls_idx = all_cls_idx[:,1:].flatten()
    # sort in decending order
    sorted_idx = np.argsort(all_scores)[::-1]
    n_top = int(max(round(num_box * args.top_ratio), 1))
    top_cls= np.unique(all_cls_idx[sorted_idx[:n_top]])

    # add bonus scores
    if not os.path.isdir(args.output_track):
        os.makedirs(args.output_track)
    for track_id, track in enumerate(tracks):
        scores = track[args.score_key]
        scores[:,top_cls] += args.top_bonus
        track[args.score_key+'_mcs'] = scores
        with open(os.path.join(args.output_track,
                '{:06d}.pkl'.format(track_id)), 'wb') as f:
            cPickle.dump(track, f, cPickle.HIGHEST_PROTOCOL)
