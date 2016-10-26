#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, 'external')
from vdetlib.utils.visual import unique_colors, add_bbox
from vdetlib.utils.common import imread, imwrite, iou
from vdetlib.utils.protocol import proto_load, frame_path_at, track_box_at_frame, annot_boxes_at_frame
import cv2
import random
import numpy as np

def sample_tracks(tracks, num):
    grouped = {}
    sampled = []
    for track in tracks:
        start_frame = track[0]['frame']
        if start_frame not in grouped:
            grouped[start_frame] = []
        grouped[start_frame].append(track)
    for start_frame, group_tracks in grouped.iteritems():
        sampled.extend(random.sample(group_tracks, num))
    return sampled

def positive_tracks(tracks, annot_proto, overlap_thres, box_key):
    grouped = {}
    sampled = []
    for track in tracks:
        start_frame = track[0]['frame']
        if start_frame not in grouped:
            grouped[start_frame] = []
        grouped[start_frame].append(track)
    for start_frame, group_tracks in grouped.iteritems():
        init_boxes = [track[0][box_key] for track in group_tracks]
        gt_boxes = annot_boxes_at_frame(annot_proto, start_frame)
        overlaps = iou(init_boxes, gt_boxes)
        max_overlaps = np.max(overlaps, axis=1)
        sample_idx = np.where(max_overlaps > overlap_thres)[0]
        sampled.extend([group_tracks[i] for i in sample_idx])
    return sampled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('track_file')
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--box_key', default='bbox',
        help='Key of bbox in the track protocol. [bbox]')
    gp1 = parser.add_argument_group('GT overlap sampling')
    gp1.add_argument('--annot_file', default=None)
    gp1.add_argument('--overlap', type=float, default=0.5)
    gp2 = parser.add_argument_group('Random sampling.')
    gp2.add_argument('--sample_tracks', action='store_true')
    gp2.add_argument('--num_tracks', type=int, default='30')
    gp2.set_defaults(sample_tracks=False)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    track_proto = proto_load(args.track_file)

    colors = random.shuffle(unique_colors(len(track_proto['tracks'])))

    if not args.save_dir:
        cv2.namedWindow('tracks')
    if args.sample_tracks:
        tracks = sample_tracks(track_proto['tracks'], args.num_tracks)
    elif args.annot_file:
        annot_proto = proto_load(args.annot_file)
        tracks = positive_tracks(track_proto['tracks'], annot_proto,
            args.overlap, args.box_key)
    else:
        tracks = track_proto['tracks']
    random.shuffle(tracks)
    for frame in vid_proto['frames']:
        img = imread(frame_path_at(vid_proto, frame['frame']))
        boxes = [track_box_at_frame(tracklet, frame['frame'], args.box_key) \
                for tracklet in tracks]
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
