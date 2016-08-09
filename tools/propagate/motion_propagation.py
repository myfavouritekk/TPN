#!/usr/bin/env python

import argparse
import glob
import os
import scipy.io as sio
from vdetlib.utils.protocol import proto_load
from vdetlib.utils.common import imread
import numpy as np
import pdb

def _boxes_average_sum(motionmap, boxes, box_ratio=1.0):
    h, w = motionmap.shape
    accum_map = np.cumsum(np.cumsum(motionmap, axis=0), axis=1)
    boxes = boxes - 1

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
    col_values = accum_map[row2.astype('int'), col1.astype('int')]
    row_values = accum_map[row1.astype('int'), col2.astype('int')]
    corner_values = accum_map[row1.astype('int'), col1.astype('int')]

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
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('det_dir')
    parser.add_argument('flow_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--window', default=3, type=int)
    args = parser.parse_args()

    window = args.window
    assert window > 0 and window % 2 == 1
    half_tws = (window - 1) / 2

    vid_proto = proto_load(args.vid_file)
    det_files = sorted(glob.glob(os.path.join(args.det_dir, '*.mat')))
    n_frames = len(vid_proto['frames'])
    assert len(vid_proto['frames']) == len(det_files)
    all_boxes = []
    all_scores = []
    num_boxes_before = 0
    num_expected = 0
    for idx, det_file in enumerate(det_files):
        det = sio.loadmat(det_file)
        all_boxes.append(det['boxes'])
        all_scores.append(det['zs'])
        num_cur_boxes = det['boxes'].shape[0]
        num_boxes_before += num_cur_boxes
        num_expected += num_cur_boxes
        num_expected += min(idx, half_tws) * num_cur_boxes
        num_expected += min(n_frames - idx - 1, half_tws) * num_cur_boxes


    # propagation
    for local_idx, (frame, det_file) in \
            enumerate(zip(vid_proto['frames'], det_files)):
        det_file_name = os.path.splitext(os.path.basename(det_file))[0]
        assert os.path.splitext(frame['path'])[0] == det_file_name
        flow_file = os.path.join(args.flow_dir,
            '{}.png'.format(det_file_name))

        print "Propagating frame {}: {}".format(frame['frame'], frame['path'])
        det = sio.loadmat(det_file)
        # read optical flows
        # rgb is reversed to bgr when using opencv
        optflow = imread(flow_file)[:,:,::-1]
        x_map = optflow_transform(optflow[:,:,0])
        y_map = optflow_transform(optflow[:,:,1])
        n_row, n_col = x_map.shape
        # read detections
        num_boxes = det['boxes'].shape[0]
        boxes = det['boxes'].reshape((-1, 4))
        box_avg_x = _boxes_average_sum(x_map, boxes)
        box_avg_x = box_avg_x.reshape((num_boxes, 1))
        box_avg_y = _boxes_average_sum(y_map, boxes)
        box_avg_y = box_avg_y.reshape((num_boxes, 1))
        motion_shift = np.concatenate(
            (box_avg_x, box_avg_y, box_avg_x, box_avg_y), axis=1)

        # motion propagation
        for offset in range(-half_tws, half_tws+1):
            if offset == 0: continue
            neighbor_frame_idx = local_idx + offset
            if neighbor_frame_idx < 0 or neighbor_frame_idx >= n_frames:
                continue

            cur_boxes = det['boxes']
            cur_scores = det['zs']
            cur_boxes = cur_boxes + motion_shift * offset

            # clipping
            cur_boxes = np.clip(cur_boxes, 1,
                np.array([n_col,n_row,n_col,n_row]).reshape((1, 1, 4)))

            all_boxes[neighbor_frame_idx] = \
                np.concatenate((all_boxes[neighbor_frame_idx], cur_boxes), axis=0)
            all_scores[neighbor_frame_idx] = \
                np.concatenate((all_scores[neighbor_frame_idx], cur_scores), axis=0)

    num_boxes_after = 0
    for box in all_boxes:
        num_boxes_after += box.shape[0]
    print "Originally {} boxes, expected {} boxes, now {} boxes.".format(
        num_boxes_before, num_expected, num_boxes_after)

    # save results
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    print "Saving..."
    for frame, boxes, scores in zip(vid_proto['frames'], all_boxes, all_scores):
        frame_name = os.path.splitext(frame['path'])[0]
        sio.savemat(os.path.join(args.save_dir, frame_name+'.mat'),
            {'boxes': boxes, 'zs': scores}, do_compression=True)

