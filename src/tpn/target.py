#!/usr/bin/env python
import numpy as np
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

def add_track_targets(track_proto, annot_proto):

    # process annot_proto
    print "Process annotation file for {}.".format(annot_proto['video'])
    processed_annot = {}
    num_gt = len(annot_proto['annotations'])
    max_gt_frame = max([int(annot['track'][-1]['frame']) for annot in annot_proto['annotations']])
    # initialize fields
    processed_annot['exist'] = np.zeros((num_gt, max_gt_frame), dtype=np.float)
    processed_annot['bbox'] = np.zeros((num_gt, 4, max_gt_frame), dtype=np.float)
    processed_annot['class_index'] = np.zeros((num_gt, max_gt_frame), dtype=np.float)
    processed_annot['occluded'] = np.zeros((num_gt, max_gt_frame), dtype=np.float)

    # fill annotations
    for track_id, annot in enumerate(annot_proto['annotations']):
        for box in annot['track']:
            frame_index = box['frame'] - 1
            processed_annot['exist'][track_id, frame_index] = 1
            processed_annot['bbox'][track_id, :, frame_index] = box['bbox']
            processed_annot['class_index'][track_id, frame_index] = box['class_index']
            processed_annot['occluded'][track_id, frame_index] = box['occluded']

    # add track targets
    print "Adding tracking targets..."
    for track in track_proto['tracks']:
        target_obj = -1 # uncertain target
        class_label = -1
        for box in track:
            frame_index = box['frame'] - 1
            if frame_index >= max_gt_frame: # no gt as the end of video
                box['class_label'] = 0
                box['bbox_target'] = [0, 0, 0, 0]
                continue
            roi = box['roi']
            overlaps = bbox_overlaps(
                np.asarray(roi, dtype=np.float)[np.newaxis,:],
                processed_annot['bbox'][:,:,frame_index])
            max_obj = np.argmax(overlaps)
            max_overlap = np.max(overlaps)

            if target_obj == -1:
                # assign target object
                if max_overlap >= 0.5:
                    target_obj = max_obj
                    class_label = processed_annot['class_index'][target_obj, frame_index]
                    assert class_label != 0
                else:
                    # still on background
                    box['class_label'] = 0
                    box['bbox_target'] = [0, 0, 0, 0]
                    continue
            assert target_obj != -1
            # if target disappears
            if processed_annot['exist'][target_obj, frame_index] == 0:
                box['class_label'] = -1
                box['bbox_target'] = [0, 0, 0, 0]
                continue

            # target still exists
            target_box = processed_annot['bbox'][target_obj, :, frame_index]
            box['bbox_target'] = bbox_transform(
                np.asarray(roi, dtype=np.float)[np.newaxis,:],
                target_box[np.newaxis, :]).tolist()

            target_overlap = overlaps[0,target_obj]
            if target_overlap >= 0.5: # still on target
                box['class_label'] = int(class_label)
            elif max_overlap >= 0.5: # drift to other objects
                box['class_label'] = -1 # ignore class label
            else: # drift to background
                box['class_label'] = 0

        # Finnaly assign ending label
        # criteria: true if
        #   1. never find target box
        #   2. drift to other object and never come back
        #   3. drift to background and never come back
        comeback = False
        for back_ind, box in enumerate(track[::-1]):
            if target_obj == -1:
                box['end_label'] = 1
                continue
            assert class_label != -1
            assert class_label != 0
            if comeback or back_ind == 0:
                box['end_label'] = 0
            else:
                box['end_label'] = 1
            if box['class_label'] == class_label:
                comeback = True

