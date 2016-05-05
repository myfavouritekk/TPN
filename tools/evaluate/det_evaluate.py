#!/usr/bin/env python

import argparse
import os
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.cython_nms import vid_nms
import multiprocessing as mp
import glob

def image_name_at_frame(vid_proto, frame_idx):
    vid_name = vid_proto['video']
    for frame in vid_proto['frames']:
        if frame['frame'] == frame_idx:
            return os.path.join(vid_name, os.path.splitext(frame['path'])[0])

def single_vid_eval(input_list):
    vid_file, det_file, image_set = input_list
    vid_proto = proto_load(vid_file)
    det_proto = proto_load(det_file)
    vid_name = vid_proto['video']
    assert vid_name == det_proto['video']

    # dets are [frame_idx, class_index, cls_score, x1, y1, x2, y2]
    dets = []
    for det in det_proto['detections']:
        local_idx = det['frame']
        image_name = image_name_at_frame(vid_proto, local_idx)
        frame_idx = image_set[image_name]
        bbox = map(lambda x:max(x,0), det['bbox'])
        for cls_score in det['scores']:
            dets.append([int(frame_idx), cls_score['class_index'],
                cls_score['score'], bbox])

    if len(dets) == 0:
        print "det_proto {} has no detections.".format(det_file)
        return dets

    nms_boxes = [[det[0],]+det[-1]+[det[2],] for det in dets]
    keep = vid_nms(np.asarray(nms_boxes).astype('float32'), args.nms_thres)

    kept_dets = [dets[i] for i in keep]
    kept_dets.sort(key=lambda x:(x[0], x[1], -x[2]))
    return kept_dets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_root')
    parser.add_argument('det_root')
    parser.add_argument('image_set_file')
    parser.add_argument('save_file')
    parser.add_argument('--nms', dest='nms_thres', default=0.5)
    parser.add_argument('--pool', dest='pool_size',
        default=mp.cpu_count())
    args = parser.parse_args()

    with open(args.image_set_file) as f:
        image_set = dict([line.strip().split() for line in f.readlines()])
    fp = open(args.save_file, 'w')

    vidfiles = glob.glob(os.path.join(args.vid_root, '*.vid'))
    det_files = [os.path.join(args.raw_det_root,
    '{}.det'.format(os.path.splitext(os.path.basename(vid_file))[0])) \
        for vid_file in vidfiles]

    input_list = []
    for vid_file, det_file in zip(vidfiles, detfiles):
        input_list.append((vid_file, det_file, image_set))

    print "Evaluating {} dets...".format(len(input_list))
    pool = mp.Pool(args.pool_size)
    dets = pool.map(single_vid_eval, input_list)

    # flatten
    kept_dets = [det for vid_det in dets for det in vid_det]
    print "Writing to {}...".format(args.save_file)
    for frame_idx, class_index, score, bbox in kept_dets:
        fp.write('{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
            frame_idx, class_index, score,
            bbox[0], bbox[1], bbox[2], bbox[3]))

