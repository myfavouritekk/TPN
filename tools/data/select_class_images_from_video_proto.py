#!/usr/bin/env python

import sys
import os.path as osp
import os
import argparse
this_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(this_dir, '../../external/'))
from vdetlib.utils.protocol import proto_load, frame_path_at, annots_at_frame
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_proto')
    parser.add_argument('annot_proto')
    parser.add_argument('save_dir')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_proto)
    annot_proto = proto_load(args.annot_proto)
    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        image_path = frame_path_at(vid_proto, frame_id)
        annots = annots_at_frame(annot_proto, frame_id)
        cls_idx = [annot['class_index'] for annot in annots]
        uniq_cls = set(cls_idx)
        for cls in uniq_cls:
            save_dir = osp.join(args.save_dir,
                "{:02d}".format(cls))
            if not osp.isdir(save_dir):
                os.makedirs(save_dir)
            save_path = osp.join(save_dir,
                '_'.join(image_path.split('/')[-2:]))
            shutil.copyfile(image_path, save_path)
