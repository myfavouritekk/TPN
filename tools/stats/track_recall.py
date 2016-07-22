#!/usr/bin/env python

from vdetlib.utils.protocol import proto_load, proto_dump, track_box_at_frame
from vdetlib.utils.common import iou
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('annot_file')
    parser.add_argument('track_file')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    annot_proto = proto_load(args.annot_file)
    track_proto = proto_load(args.track_file)

    gt_count = 0
    recall_count = 0
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        # annot boxes
        annot_boxes = [track_box_at_frame(annot_track['track'], frame_id) \
            for annot_track in annot_proto['annotations']]
        annot_boxes = [box for box in annot_boxes if box is not None]

        if len(annot_boxes) == 0: continue
        gt_count += len(annot_boxes)

        # track boxes
        track_boxes = [track_box_at_frame(tracklet, frame_id) \
            for tracklet in track_proto['tracks']]
        track_boxes = [box for box in track_boxes if box is not None]
        if len(track_boxes) == 0: continue

        overlaps = iou(np.asarray(annot_boxes), np.asarray(track_boxes))
        max_overlaps = overlaps.max(axis=1)
        recall_count += np.count_nonzero(max_overlaps >= 0.5)

    print "{} {} {} {}".format(vid_proto['video'],
        gt_count, recall_count, float(recall_count) / gt_count)
