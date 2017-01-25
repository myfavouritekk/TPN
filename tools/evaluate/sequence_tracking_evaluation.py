#!/usr/bin/env python

import argparse
import sys
import os.path as osp
import numpy as np
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/'))
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.common import iou
sys.path.insert(0, osp.join(this_dir, '../../external/py-faster-rcnn/lib'))
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv

def annots_at_frame(annot_proto, frame_id):
    annots = []
    for annot in annot_proto['annotations']:
        for box in annot['track']:
            if box['frame'] == frame_id:
                annots.append((box, annot['id']))
    return annots

def annot_by_id(annot_proto, annot_id):
    for annot in annot_proto['annotations']:
        if annot['id'] == annot_id:
            return annot
    return None

def select_gt_segment(gt, st, end):
    boxes = []
    for box in gt:
        frame_id = box['frame']
        if frame_id >= st:
            boxes.append(box['bbox'])
        if frame_id >= end:
            break
    return boxes

def _accuracy(track, gt):
    if len(track) < 2:
        return [], []
    abs_acc = []
    rel_acc = []
    ious = []
    st_frame = track[0]['frame']
    end_frame = track[-1]['frame']
    assert end_frame - st_frame + 1 == len(track)
    gt_seg = select_gt_segment(gt['track'], st_frame, end_frame)
    assert len(gt_seg) <= len(track)
    track_bbox1 = np.asarray([track[0]['bbox']])
    gt_bbox1 = np.asarray([gt_seg[0]])
    for track_box, gt_bbox in zip(track[1:len(gt_seg)], gt_seg[1:]):
        # current track box
        track_bbox = np.asarray([track_box['bbox']])
        # gt motion
        gt_delta = bbox_transform(gt_bbox1, np.asarray([gt_bbox]))
        # target is the first track_bbox with gt motion
        track_bbox_target = bbox_transform_inv(track_bbox1, gt_delta)
        abs_diff = np.abs(track_bbox - track_bbox_target)
        cur_iou = iou(track_bbox, track_bbox_target)
        width = track_bbox_target[0,2] - track_bbox_target[0,0]
        height = track_bbox_target[0,3] - track_bbox_target[0,1]
        rel_diff = abs_diff / (np.asarray([width, height, width, height]) + np.finfo(float).eps)
        abs_acc.extend(abs_diff.flatten().tolist())
        rel_acc.extend(rel_diff.flatten().tolist())
        ious.extend(cur_iou.flatten().tolist())
    return abs_acc, rel_acc, ious


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('track_file')
    parser.add_argument('annot_file')
    parser.add_argument('--overlap', type=float, default=0.5,
        help='GT overlap for considering positive samples.')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    track_proto = proto_load(args.track_file)
    annot_proto = proto_load(args.annot_file)

    acc = {'abs_acc':[], 'rel_acc':[], 'ious':[]}
    for track in track_proto['tracks']:
        frame1_id = track[0]['frame']
        annots = annots_at_frame(annot_proto, frame1_id)
        annot_boxes = [annot[0]['bbox'] for annot in annots]
        if len(annot_boxes) == 0:
            continue
        gt_overlaps = iou([track[0]['roi']], annot_boxes)
        max_overlap = np.max(gt_overlaps, axis=1)
        if max_overlap < args.overlap: continue
        max_gt = np.argmax(gt_overlaps, axis=1)[0]
        gt_idx = annots[max_gt][1]
        gt_annot = annot_by_id(annot_proto, gt_idx)
        abs_acc, rel_acc, ious = _accuracy(track, gt_annot)
        acc['abs_acc'].extend(abs_acc)
        acc['rel_acc'].extend(rel_acc)
        acc['ious'].extend(ious)
    print "{}: abs_diff {:.06f} relative_diff {:.06f} mean IOU: {:.06f}".format(
        vid_proto['video'], np.mean(acc['abs_acc']),
        np.mean(acc['rel_acc']), np.mean(acc['ious']))
