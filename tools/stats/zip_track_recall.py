#!/usr/bin/env python

from vdetlib.utils.protocol import proto_load, proto_dump, track_box_at_frame
from vdetlib.utils.common import iou
import argparse
import numpy as np
import glob
import cPickle
import zipfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('annot_file')
    parser.add_argument('track_dir')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    annot_proto = proto_load(args.annot_file)
    tracks = []
    frames = []
    if zipfile.is_zipfile(args.track_dir):
        zf = zipfile.ZipFile(args.track_dir)
        track_files = zf.namelist()
        for track_file in track_files:
            track = cPickle.loads(zf.read(track_file))
            tracks.append(track['bbox'])
            frames.append(track['frame'])
    else:
        track_files = glob.glob(args.track_dir + "/*.pkl")
        for track_file in track_files:
            track = cPickle.loads(open(track_file, 'rb').read())
            tracks.append(track['bbox'])
            frames.append(track['frame'])

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
        track_boxes = [track[frame==frame_id,:].flatten() for track, frame \
            in zip(tracks, frames) if np.any(frame==frame_id)]
        if len(track_boxes) == 0: continue

        overlaps = iou(np.asarray(annot_boxes), np.asarray(track_boxes))
        max_overlaps = overlaps.max(axis=1)
        recall_count += np.count_nonzero(max_overlaps >= 0.5)

    print "{} {} {} {}".format(vid_proto['video'],
        gt_count, recall_count, float(recall_count) / gt_count)
