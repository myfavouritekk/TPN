#!/usr/bin/env python
# propagate bounding boxes

from fast_rcnn.test import im_detect
from vdetlib.utils.protocol import proto_load, proto_dump, frame_path_at
from vdetlib.utils.timer import Timer
from vdetlib.utils.common import imread
import numpy as np

def _append_boxes(tracks, frame_id, boxes, scores):
    if not tracks:
        # init tracks
        for _ in boxes:
            tracks.append([])
    for track, bbox in zip(tracks, boxes):
        track.append({
            "frame": frame_id,
            "bbox": bbox.tolist(),
            "anchor": frame_id - 1
        })

def naive_box_regression(net_rpn, net_no_rpn, vid_proto,
        scheme='max', class_idx=None):
    """Generating tubelet proposals based on the region proposals of first frame."""

    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'naive_box_regression'
    tracks = []
    pred_boxes = None

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        if idx == 1:
            scores, boxes = im_detect(net_rpn, im, pred_boxes)
        else:
            scores, boxes = im_detect(net_no_rpn, im, pred_boxes)

        boxes = boxes.reshape((boxes.shape[0], -1, 4))
        if scheme is 'mean' or idx == 1:
            # use mean regressions as predictios
            pred_boxes = np.mean(boxes, axis=1)
        elif scheme is 'max':
            # use the regressions of the class with the maximum probability
            # excluding __background__ class
            max_cls = scores[:,1:].argmax(axis=1) + 1
            pred_boxes = boxes[np.arange(len(boxes)), max_cls, :]
        else:
            # use class specific regression as predictions
            pred_boxes = boxes[:,class_idx,:]
        _append_boxes(tracks, frame['frame'], pred_boxes, scores)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    track_proto['tracks'] = tracks
    return track_proto


def _box_proto_to_track(box_proto, max_frame, length):
    # generate empty tracks according to box proto
    tracks = []
    for box in box_proto['boxes']:
        track = []
        for i in xrange(length):
            if i == 0:
                track_box = {
                    "frame": box['frame'] + i,
                    "roi": box['bbox'],
                    "anchor": i
                }
            else:
                track_box = {
                    "frame": box['frame'] + i,
                    "anchor": i
                }
            track.append(track_box)
        tracks.append(track)
    return tracks

def _cur_rois(tracks, frame_id):
    rois = []
    index = []
    for track_id, track in enumerate(tracks):
        for box in track:
            if box['frame'] == frame_id:
                rois.append(box['roi'])
                index.append(track_id)
                break
    return rois, index

def _update_track(tracks, pred_boxes, track_index, frame_id):
    for i, pred_box in zip(track_index, pred_boxes):
        for box in tracks[i]:
            if box['frame'] == frame_id:
                box['bbox'] = pred_box.tolist()
            if box['frame'] == frame_id + 1:
                box['roi'] = pred_box.tolist()
                break

def roi_propagation(vid_proto, box_proto, net, scheme='max', length=None):
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'roi_propagation'
    max_frame = vid_proto['frames'][-1]['frame']
    if not length: length = max_frame
    tracks = _box_proto_to_track(box_proto, max_frame, length)
    batch_size = 1024

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois, track_index = _cur_rois(tracks, frame['frame'])
        # print "Frame {}: {} proposals".format(frame['frame'], len(rois))
        timer = Timer()
        timer.tic()
        # scores: n x c, boxes: n x (c x 4)
        scores = []
        boxes = []
        for roi_batch in np.split(np.asarray(rois), range(0, len(rois), batch_size)[1:]):
            s_batch, b_batch = im_detect(net, im, np.asarray(roi_batch))
            scores.append(s_batch)
            boxes.append(b_batch)
        scores = np.concatenate(scores, axis=0)
        boxes = np.concatenate(boxes, axis=0)
        boxes = boxes.reshape((boxes.shape[0], -1, 4))

        if scheme is 'mean':
            # use mean regressions as predictios
            pred_boxes = np.mean(boxes, axis=1)
        elif scheme is 'max':
            # use the regressions of the class with the maximum probability
            # excluding __background__ class
            max_cls = scores[:,1:].argmax(axis=1) + 1
            pred_boxes = boxes[np.arange(len(boxes)), max_cls, :]
        elif scheme is 'weighted':
            # use class specific regression as predictions
            pred_boxes = np.sum(boxes * scores[:,:,np.newaxis], axis=1) / np.sum(scores, axis=1, keepdims=True)
        _update_track(tracks, pred_boxes, track_index, frame['frame'])
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, len(rois))
    track_proto['tracks'] = tracks
    return track_proto
