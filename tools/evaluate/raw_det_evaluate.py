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

def single_vid_raw_det_eval(input_list, thresh=0.00, max_per_image=100, nms_thres=0.3,
        num_classes=31):
    vid_file, det_folder, image_set = input_list
    vid_proto = proto_load(vid_file)
    print vid_proto['video']

    det_strings = []
    for frame in vid_proto['frames']:
        frame_name = os.path.splitext(frame['path'])[0]
        det_file = os.path.join(det_folder, "{}.mat".format(frame_name))
        local_idx = frame['frame']
        image_name = image_name_at_frame(vid_proto, local_idx)
        frame_idx = image_set[image_name]
        try:
            det = sio.loadmat(det_file)
        except Exception as error:
            print "Error {}: det_file {}.".format(error, det_file)
            continue
        boxes = det['boxes']
        scores = det['zs']
        cur_boxes = [[] for _ in xrange(num_classes)]
        for j in xrange(1, num_classes):
            inds = np.where(scores[:,j] > thresh)[0]
            if len(inds) == 0: continue
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j, :]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, nms_thres)
            cls_dets = cls_dets[keep, :]
            cur_boxes[j] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([cur_boxes[j][:, -1] \
                for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(cur_boxes[j][:, -1] >= image_thresh)[0]
                    cur_boxes[j] = cur_boxes[j][keep, :]

        for class_index, cls_dets in enumerate(cur_boxes):
            if class_index == 0: continue
            for dets in cls_dets:
                det_strings.append('{} {} {:.06f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(
                    frame_idx, class_index, dets[-1],
                    dets[0], dets[1], dets[2], dets[3]))

    return det_strings

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

    vidfiles = sorted(glob.glob(os.path.join(args.vid_root, '*.vid')))
    det_folders = [os.path.join(args.raw_det_root,
        '{}'.format(os.path.splitext(os.path.basename(vid_file))[0])) \
            for vid_file in vidfiles]
    input_list = []
    for vid_file, det_folder in zip(vidfiles, det_folders):
        input_list.append((vid_file, det_folder, image_set))

    print "Evaluating {} dets...".format(len(input_list))
    pool = mp.Pool(args.pool_size)
    det_strings = pool.map(single_vid_raw_det_eval, input_list)
    # det_strings = map(single_vid_raw_det_eval, input_list)

    # flatten
    print "Writing to {}...".format(args.save_file)
    for vid_strings in det_strings:
        for string in vid_strings:
            fp.write(string)

