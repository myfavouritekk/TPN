#!/usr/bin/env python

import argparse
import os
import os.path as osp
import sys
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external'))
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
from vdetlib.utils.visual import unique_colors, add_bbox
from vdetlib.utils.common import imread, imwrite
from vdetlib.utils.protocol import proto_load, frame_path_at, boxes_at_frame, annot_boxes_at_frame
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps
import cv2
import random
import numpy as np

def _sample_boxes(box_proto, frame_id, num, annot_proto=None):
    boxes = boxes_at_frame(box_proto, frame_id)
    boxes = [box['bbox'] for box in boxes]
    if annot_proto is None:
        boxes = random.sample(boxes, num)
    else:
        gt_boxes = annot_boxes_at_frame(annot_proto, frame_id)
        overlaps = bbox_overlaps(np.asarray(boxes, dtype=np.float),
                                 np.asarray(gt_boxes, dtype=np.float))
        max_overlaps = np.max(overlaps, axis=1)
        idx = np.argsort(max_overlaps)[::-1][:num]
        boxes = [boxes[i] for i in idx]
    return boxes

def _propagate_boxes(boxes, annot_proto, frame_id):
    pred_boxes = []
    annots = []
    for annot in annot_proto['annotations']:
        for idx, box in enumerate(annot['track']):
            if box['frame'] == frame_id and len(annot['track']) > idx + 1:
                gt1 = box['bbox']
                gt2 = annot['track'][idx+1]['bbox']
                delta = bbox_transform(np.asarray([gt1]), np.asarray([gt2]))
                annots.append((gt1, delta))
    gt1 = [annot[0] for annot in annots]
    overlaps = bbox_overlaps(np.require(boxes, dtype=np.float),
                             np.require(gt1, dtype=np.float))
    assert len(overlaps) == len(boxes)
    for gt_overlaps, box in zip(overlaps, boxes):
        max_overlap = np.max(gt_overlaps)
        max_gt = np.argmax(gt_overlaps)
        if max_overlap < 0.5:
            pred_boxes.append(box)
        else:
            delta = annots[max_gt][1]
            pred_boxes.append(bbox_transform_inv(np.asarray([box]), delta)[0].tolist())
    return pred_boxes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('box_file')
    parser.add_argument('gt_file')
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--num_tracks', type=int, default=30)
    parser.add_argument('--length', type=int, default=20)
    parser.set_defaults(sample_tracks=False)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    box_proto = proto_load(args.box_file)
    annot_proto = proto_load(args.gt_file)

    paried_frames = zip(vid_proto['frames'][:-1], vid_proto['frames'][1:])
    # colors = unique_colors(len(track_proto['tracks']))

    boxes = []
    colors = None
    for frame1, frame2 in paried_frames:
        frame_id = frame1['frame']
        img = imread(frame_path_at(vid_proto, frame1['frame']))
        if frame_id % args.length == 1 or len(boxes) == 0:
            boxes = _sample_boxes(box_proto, frame_id, args.num_tracks, annot_proto)
            colors = unique_colors(len(boxes))
        else:
            boxes = _propagate_boxes(boxes, annot_proto, frame_id-1)
        tracked = add_bbox(img, boxes, None, colors, 2)
        if args.save_dir:
            if not os.path.isdir(args.save_dir):
                try:
                    os.makedirs(args.save_dir)
                except:
                    pass
            imwrite(os.path.join(args.save_dir, "{:04d}.jpg".format(frame['frame'])),
                    tracked)
        else:
            cv2.imshow('tracks', tracked)
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit(0)
    if not args.save_dir:
        cv2.destroyAllWindows()
