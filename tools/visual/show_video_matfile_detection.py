#!/usr/bin/env python

import argparse
import os
import sys
import cv2
from vdetlib.vdet.dataset import imagenet_vdet_class_idx
from vdetlib.utils.common import imread, imwrite
from vdetlib.utils.protocol import proto_load, proto_dump, frame_path_at, frame_top_detections
from vdetlib.utils.visual import add_bbox
from vdetlib.utils.cython_nms import nms
import scipy.io as sio
import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('det_root')
    parser.add_argument('--cls', choices=imagenet_vdet_class_idx.keys())
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--top_k', default=10)
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    det_files = glob.glob(os.path.join(args.det_root, '*.mat'))
    if args.save_dir and not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    cls_index = imagenet_vdet_class_idx[args.cls]

    for frame, det_file in zip(vid_proto['frames'], det_files):
        det = sio.loadmat(det_file)
        frame_idx = frame['frame']
        img = imread(frame_path_at(vid_proto, frame_idx))
        boxes = det['boxes'][:,cls_index,:]
        scores = det['zs'][:,cls_index]
        keep = nms(np.hstack((boxes,scores[:,np.newaxis])), 0.3)
        det_img = add_bbox(img, [boxes[i,:] for i in keep])
        cv2.imshow('detection', det_img)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        cv2.destroyAllWindows()
        if args.save_dir:
            imwrite(os.path.join(args.save_dir, "{:04d}.jpg".format(frame_idx)), det_img)

