#!/usr/bin/env python

import argparse
import glob
import os
import scipy.io as sio
from vdetlib.utils.common import imread
import numpy as np
import pdb
import cPickle
import time

def _boxes_average_sum(motionmap, boxes, box_ratio=1.0):
    h, w = motionmap.shape
    accum_map = np.cumsum(np.cumsum(motionmap, axis=0), axis=1)
    boxes = np.around(boxes)

    col1 = boxes[:,0]
    row1 = boxes[:,1]
    col2 = boxes[:,2]
    row2 = boxes[:,3]

    n_row = row2 - row1 + 1
    n_col = col2 - col1 + 1

    col1 = np.round(col1 + 0.5*(1.-box_ratio)*n_col)
    row1 = np.round(row1 + 0.5*(1.-box_ratio)*n_row)
    col2 = np.round(col2 - 0.5*(1.-box_ratio)*n_col)
    row2 = np.round(row2 - 0.5*(1.-box_ratio)*n_row)


    # clipping
    col1[col1 < 0] = 0
    row1[row1 < 0] = 0
    col2[col2 >= w] = w-1
    row2[row2 >= h] = h-1

    n_row = row2 - row1 + 1
    n_col = col2 - col1 + 1

    # print col1, col2, row1, row2

    col_out_idx = (col1==0)
    row_out_idx = (row1==0)
    corner_out_idx = col_out_idx | row_out_idx

    col1[col_out_idx] = 0
    row1[row_out_idx] = 0

    sum_values = accum_map[row2.astype('int'), col2.astype('int')]
    corner_values = accum_map[row1.astype('int'), col1.astype('int')]
    col_values = accum_map[row2.astype('int'), col1.astype('int')]
    row_values = accum_map[row1.astype('int'), col2.astype('int')]

    corner_values[corner_out_idx] = 0
    col_values[col_out_idx] = 0
    row_values[row_out_idx] = 0

    values = sum_values - col_values - row_values + corner_values
    values = values / (n_row * n_col)
    return values

def optflow_transform(optflow):
    bound = 15
    return optflow.astype('single') / 255. * 2 * bound - bound

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Motion propagation on all detections.')
    parser.add_argument('det_file')
    parser.add_argument('save_file')
    parser.add_argument('image_list')
    parser.add_argument('flow_dir')
    parser.add_argument('--window', default=3, type=int,
        help='Propagation window size. [3]')
    parser.add_argument('--sample_rate', default=4, type=int,
        help='Sample rate for starting propagation. [4]')
    parser.add_argument('--cls', type=int, choices=range(1,31),
        help='class index')
    args = parser.parse_args()

    window = args.window
    assert window > 0 and window % 2 == 1
    half_tws = (window - 1) / 2

    with open(args.det_file, 'rb') as f:
        print "Loading detection from {}...".format(args.det_file)
        all_boxes = cPickle.load(f)
    with open(args.image_list) as f:
        print "Loading image list from {}...".format(args.image_list)
        image_list = [line.strip().split()[0] for line in f]
    new_all_boxes = []
    for cls_idx, cls_dets in enumerate(all_boxes):
        assert len(cls_dets) == len(image_list)
        if cls_idx != args.cls:
            new_all_boxes.append([])
        else:
            new_all_boxes.append([np.zeros((0, 5), dtype=np.float32) for __ in cls_dets])

    for cls_idx, cls_dets in enumerate(all_boxes):
        if cls_idx != args.cls:
            continue

        print cls_idx
        cur_vid_name = None
        start_time = time.time()
        for global_idx, (frame_name, frame_det) in \
                enumerate(zip(image_list, cls_dets)):
            # frame_name: 'ILSVRC2015_val_00177000/000066'
            vid_name, frame_idx = frame_name.split('/')
            frame_idx = int(frame_idx) # 0-based
            need_propagate = (frame_idx % args.sample_rate == 0)

            if (global_idx + 1) % 1000 == 0:
                end_time = time.time()
                print "{} frames processed: {} s".format(global_idx + 1, end_time - start_time)
                start_time = time.time()

            if not need_propagate or frame_det.shape[0] == 0: continue

            # read optical flows
            # rgb is reversed to bgr when using opencv
            flow_file = os.path.join(args.flow_dir, frame_name + '.png')
            optflow = imread(flow_file)[:,:,::-1]
            x_map = optflow_transform(optflow[:,:,0])
            y_map = optflow_transform(optflow[:,:,1])
            n_row, n_col = x_map.shape

            # compute motion shift
            boxes = frame_det[:,:4]
            scores = frame_det[:,[4]]
            num_boxes = boxes.shape[0]
            box_avg_x = _boxes_average_sum(x_map, boxes)
            box_avg_x = box_avg_x.reshape((num_boxes, 1))
            box_avg_y = _boxes_average_sum(y_map, boxes)
            box_avg_y = box_avg_y.reshape((num_boxes, 1))
            motion_shift = np.concatenate(
                (box_avg_x, box_avg_y, box_avg_x, box_avg_y), axis=1)

            for offset in xrange(0, args.window+1):
                neighbor_idx = global_idx + offset
                if neighbor_idx >= len(image_list): break
                neighbor_frame_name = image_list[neighbor_idx]
                neighbor_vid_name = neighbor_frame_name.split('/')[0]
                if neighbor_vid_name != vid_name: break

                cur_boxes = boxes + motion_shift * offset
                cur_dets = np.concatenate((cur_boxes, scores), axis=1)
                new_all_boxes[cls_idx][neighbor_idx] = \
                    np.concatenate((new_all_boxes[cls_idx][neighbor_idx], cur_dets), axis=0)

    # save results
    save_dir = os.path.dirname(args.save_file)
    if save_dir and not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(args.save_file, 'wb') as f:
        cPickle.dump(new_all_boxes, f, cPickle.HIGHEST_PROTOCOL)

