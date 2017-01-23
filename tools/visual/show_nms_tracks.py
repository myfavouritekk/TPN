#!/usr/bin/env python

import argparse
import os
import os.path as osp
import glob
import numpy as np
import cv2
import sys
this_dir=osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
sys.path.insert(0, osp.join(this_dir, '../../src'))
from fast_rcnn.nms_wrapper import nms
import cPickle
from time import time
from tpn.data_io import tpn_test_iterator
sys.path.insert(0, osp.join(this_dir, '../../external'))
from vdetlib.utils.visual import unique_colors, add_bbox
from vdetlib.utils.common import imread, imwrite
from vdetlib.utils.protocol import frame_path_at, proto_load
from vdetlib.vdet.dataset import imagenet_vdet_classes

def _frame_dets(tracks, frame_idx, score_key, box_key):
    scores = []
    boxes = []
    track_ids = []
    for track_idx, track in enumerate(tracks):
        if frame_idx not in track['frame']: continue
        assert score_key in track
        assert box_key in track
        ind = track['frame'] == frame_idx
        cur_scores = track[score_key][ind]
        cur_boxes = track[box_key][ind,:]
        num_cls = cur_scores.shape[1]
        # repeat boxes if not class specific
        if cur_boxes.shape[1] != num_cls:
            cur_boxes = np.repeat(cur_boxes[:,np.newaxis,:], num_cls, axis=1)
        track_ids.append(track_idx)
        scores.append(cur_scores)
        boxes.append(cur_boxes)
    scores = np.concatenate(scores, 0)
    boxes = np.concatenate(boxes, 0)
    return track_ids, scores, boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('track_file',
        help='Track file.')
    parser.add_argument('score_key')
    parser.add_argument('box_key')
    parser.add_argument('--thres', type=float, default=0.1,
        help='Detection score threshold. [0.1]')
    parser.add_argument('--num_classes', type=int, default=31,
        help='Number of classes. [31]')
    parser.add_argument('--max_per_image', type=int, default=100,
        help='Maximum number of detections per image. [100]')
    parser.add_argument('--save_dir', 
        help='Save directory.')
    args = parser.parse_args()

    num_classes = args.num_classes

    # process vid detections
    vid_proto = proto_load(args.vid_file)
    tracks = tpn_test_iterator(args.track_file)
    end_frame_idx = set([track['frame'][-1] for track in tracks])
    end_frames = [frame for frame in vid_proto['frames'] \
        if frame['frame'] in end_frame_idx]
    vid_name = vid_proto['video']
    kept_track_ids = []
    kept_class = []
    for frame in end_frames:
        frame_name = osp.join(vid_name, osp.splitext(frame['path'])[0])

        frame_idx = frame['frame']
        start_time = time()
        track_ids, scores, boxes = _frame_dets(tracks, frame_idx, args.score_key, args.box_key)
        boxes = boxes.reshape((boxes.shape[0], -1))

        for j in xrange(1, num_classes):
            inds = np.where(scores[:, j] > args.thres)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, 0.3, force_cpu=True)
            for keep_id in keep:
                kept_track_ids.append(track_ids[inds[keep_id]])
                kept_class.append(j)

    colors = unique_colors(len(kept_track_ids))
    kept_tracks = [tracks[i] for i in kept_track_ids]
    idx = 0
    while True:
        frame = vid_proto['frames'][idx]
        frame_id = frame['frame']
        print "Frame id: {}".format(frame_id)
        img = imread(frame_path_at(vid_proto, frame['frame']))
        boxes = []
        scores = []
        show_track_ids = []
        cur_colors = []
        cur_classes = []
        for track_id, (class_id, track) in enumerate(zip(kept_class, kept_tracks)):
            if frame_id in track['frame']:
                boxes.append(track[args.box_key][track['frame'] == frame_id][0,class_id,:].tolist())
                scores.append(track[args.score_key][track['frame'] == frame_id][0,class_id].tolist())
                cur_colors.append(colors[track_id])
                cur_classes.append(imagenet_vdet_classes[class_id])
                show_track_ids.append(track_id)
        tracked = add_bbox(img, boxes, classes=cur_classes,
                           scores=scores, line_width=10)
        if args.save_dir:
            imwrite(os.path.join(args.save_dir, "{:04d}.jpg".format(frame_id)),
                tracked)
            idx += 1
            if idx >= len(vid_proto['frames']):
                break
            continue
        cv2.imshow('tracks', tracked)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        elif key == ord('a'):
            if idx > 0:
                idx -= 1
        elif key == ord('d'):
            if idx < len(vid_proto['frames']) - 1:
                idx += 1
    if not args.save_dir:
        cv2.destroyAllWindows()

