#!/usr/bin/env python

import os, sys
import pdb
import argparse
import scipy.io as sio
import h5py
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_dump, path_to_index, proto_load

def save_if_not_exist(proto, path):
    if not os.path.isfile(path):
        proto_dump(box_proto, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proposal_file')
    parser.add_argument('vid_root')
    parser.add_argument('save_root')
    args = parser.parse_args()

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    mat_file = sio.loadmat(args.proposal_file)
    image_names = mat_file['images']
    all_boxes = mat_file['boxes']
    cur_vid_name = None
    for [[image_name]], [boxes] in zip(image_names, all_boxes):
        parts = image_name.split('/')
        if len(parts) == 3:
            subset, video_name, frame_name = parts
        elif len(parts) == 4:
            __, subset, video_name, frame_name = parts
        else:
            raise ValueError('image name has {} components: {}'.format(
                len(parts), image_name))
        # start a new video
        if cur_vid_name != video_name:
            if cur_vid_name is not None:
                print "Saving {}...".format(cur_vid_name)
                save_if_not_exist(box_proto,
                    os.path.join(args.save_root, cur_vid_name+'.box'))
            print "Processsing {}...".format(video_name)
            box_proto = {}
            box_proto['video'] = video_name
            box_proto['boxes'] = []
            cur_vid_name = video_name
            # read vid_proto
            vid_proto = proto_load(
                os.path.join(args.vid_root, cur_vid_name+'.vid'))
        # process boxes
        frame_idx = path_to_index(vid_proto, frame_name)
        for box in boxes:
            # pdb.set_trace()
            box_proto['boxes'].append(
                {
                    "frame": frame_idx,
                    "bbox": box[0:4].tolist(),
                    "positive": True if box[4] == 1 else False
                }
            )
    # save last proto
    save_if_not_exist(box_proto,
        os.path.join(args.save_root, cur_vid_name+'.box.gz'))
