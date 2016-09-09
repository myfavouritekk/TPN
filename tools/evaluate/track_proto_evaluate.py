#!/usr/bin/env python

import argparse
import os
import os.path as osp
import glob
import numpy as np
import sys
import cPickle
from time import time
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib/'))
sys.path.insert(0, osp.join(this_dir, '../../src'))
sys.path.insert(0, osp.join(this_dir, '../../external/'))
from vdetlib.utils.protocol import proto_load
from fast_rcnn.nms_wrapper import nms
from tpn.evaluate import write_ilsvrc_results_file

def _frame_dets(tracks, frame_idx, score_key, box_key):
    scores = []
    boxes = []
    for track in tracks:
        for frame in track:
            if frame_idx != frame['frame']: continue
            assert score_key in frame
            assert box_key in frame
            cur_scores = np.asarray(frame[score_key]).flatten()[np.newaxis,:]
            cur_boxes = np.asarray(frame[box_key]).flatten()[np.newaxis,:]
            num_cls = cur_scores.shape[1]
            assert cur_boxes.shape[1] in [4, 4 * num_cls]
            # repeat boxes if not class specific
            if cur_boxes.shape[1] == 4:
                cur_boxes = np.tile(cur_boxes, num_cls)
            scores.append(cur_scores.copy())
            boxes.append(cur_boxes.copy())
    scores = np.vstack(scores)
    boxes = np.vstack(boxes)
    return scores, boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('track_dir',
        help='Directory that contains all track detection results.')
    parser.add_argument('vid_dir')
    parser.add_argument('image_list')
    parser.add_argument('score_key')
    parser.add_argument('box_key')
    parser.add_argument('output_dir')
    parser.add_argument('--results', type=str, default='',
        help='Result file.')
    parser.add_argument('--thres', type=float, default=0.01,
        help='Detection score threshold. [0.01]')
    parser.add_argument('--num_classes', type=int, default=31,
        help='Number of classes. [31]')
    parser.add_argument('--max_per_image', type=int, default=100,
        help='Maximum detection in each image. [100]')
    args = parser.parse_args()

    # read image_list
    with open(args.image_list, 'r') as f:
        image_list = dict([line.strip().split() for line in f])

    num_classes = args.num_classes
    all_boxes = [[[] for _ in xrange(len(image_list))]
                 for _ in xrange(num_classes)]

    # process vid detections
    tracks = sorted(glob.glob(osp.join(args.track_dir, '*')))
    for track_path in tracks:
        print track_path
        vid_name = osp.split(track_path)[-1].split('.')[0]
        vid_proto = proto_load(osp.join(args.vid_dir, vid_name + '.vid'))
        track_proto = proto_load(track_path)
        for frame in vid_proto['frames']:
            frame_name = osp.join(vid_name, osp.splitext(frame['path'])[0])
            if frame_name not in image_list.keys(): continue

            frame_idx = frame['frame']
            sub_idx = int(image_list[frame_name])
            global_idx = sub_idx - 1
            start_time = time()
            scores, boxes = _frame_dets(track_proto['tracks'], frame_idx,
                args.score_key, args.box_key)
            boxes = boxes.reshape((boxes.shape[0], -1))

            for j in xrange(1, num_classes):
                inds = np.where(scores[:, j] > args.thres)[0]
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
                keep = nms(cls_dets, 0.3, force_cpu=True)
                cls_dets = cls_dets[keep, :]
                all_boxes[j][global_idx] = cls_dets

            # Limit to max_per_image detections *over all classes*
            if args.max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][global_idx][:, -1]
                                          for j in xrange(1, num_classes)])
                if len(image_scores) > args.max_per_image:
                    image_thresh = np.sort(image_scores)[-args.max_per_image]
                    for j in xrange(1, num_classes):
                        keep = np.where(all_boxes[j][global_idx][:, -1] >= image_thresh)[0]
                        all_boxes[j][global_idx] = all_boxes[j][global_idx][keep, :]
            end_time = time()
            print "{}/{}: {:.03f} s".format(sub_idx, len(image_list), end_time - start_time)
            sys.stdout.flush()

    det_file = osp.join(args.output_dir, 'detections.pkl')
    if not osp.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    if args.results:
        with open(args.results, 'w') as f:
            write_ilsvrc_results_file(all_boxes, f, thres=args.thres)
