#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, 'external')
from vdetlib.utils.visual import unique_colors, add_bbox
from vdetlib.utils.common import imread, imwrite
from vdetlib.utils.protocol import proto_load, frame_path_at, track_box_at_frame
import cv2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('track_file')
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--box_key', default='bbox',
        help='Key of bbox in the track protocol. [bbox]')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    track_proto = proto_load(args.track_file)

    colors = unique_colors(len(track_proto['tracks']))

    if not args.save_dir:
        cv2.namedWindow('tracks')
    for frame in vid_proto['frames']:
        img = imread(frame_path_at(vid_proto, frame['frame']))
        boxes = [track_box_at_frame(tracklet, frame['frame'], args.box_key) \
                for tracklet in track_proto['tracks']]
        tracked = add_bbox(img, boxes, None, None, 2)
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
