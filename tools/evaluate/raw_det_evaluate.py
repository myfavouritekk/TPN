#!/usr/bin/env python

import argparse
import os
import numpy as np
import sys
sys.path.insert(1, '.')
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.cython_nms import nms
import multiprocessing as mp
import glob
import scipy.io as sio

def image_name_at_frame(vid_proto, frame_idx):
    vid_name = vid_proto['video']
    for frame in vid_proto['frames']:
        if frame['frame'] == frame_idx:
            return os.path.join(vid_name, os.path.splitext(frame['path'])[0])

def single_vid_raw_det_eval(input_list):
    vid_file, det_folder, image_set = input_list
    vid_proto = proto_load(vid_file)
    print vid_proto['video']

    # dets are [frame_idx, class_index, cls_score, x1, y1, x2, y2]
    dets = []
    for frame in vid_proto['frames']:
        frame_name = os.path.splitext(frame['path'])[0]
        det_file = os.path.join(det_folder, "{}.mat".format(frame_name))
        local_idx = frame['frame']
        try:
            det = sio.loadmat(det_file)
        except ValueError:
            print "det_file {} has been corrupted.".format(det_file)
            continue
        boxes = det['boxes']
        scores = det['zs']
        image_name = image_name_at_frame(vid_proto, local_idx)
        frame_idx = image_set[image_name]
        cur_dets = []
        for reg_boxes, cls_scores in zip(boxes, scores):
            for cls_index, (cls_bbox, cls_score) in \
                enumerate(zip(reg_boxes, cls_scores)):
                # skip background
                if cls_index == 0: continue
                cur_dets.append([int(frame_idx), cls_index,
                    float(cls_score), cls_bbox.tolist()])
        nms_boxes = [det[-1]+[det[2],] for det in cur_dets]
        keep = nms(np.asarray(nms_boxes).astype('float32'), args.nms_thres)
        for i in keep:
            dets.append(cur_dets[i])

    dets.sort(key=lambda x:(x[0], x[1], -x[2]))
    return dets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_root')
    parser.add_argument('raw_det_root')
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
    det_folders = [os.path.join(args.raw_det_root,
        '{}'.format(os.path.splitext(os.path.basename(vid_file))[0])) \
            for vid_file in vidfiles]
    input_list = []
    for vid_file, det_folder in zip(vidfiles, det_folders):
        input_list.append((vid_file, det_folder, image_set))

    print "Evaluating {} dets...".format(len(input_list))
    pool = mp.Pool(args.pool_size)
    dets = pool.map(single_vid_raw_det_eval, input_list)

    # flatten
    kept_dets = [det for vid_det in dets for det in vid_det]
    print "Writing to {}...".format(args.save_file)
    for frame_idx, class_index, score, bbox in kept_dets:
        fp.write('{} {} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
            frame_idx, class_index, score,
            bbox[0], bbox[1], bbox[2], bbox[3]))

