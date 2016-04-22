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

def naive_box_regression(net_rpn, net_no_rpn, vid_proto):
    """Generating tubelet proposals based on the region proposals of first frame."""

    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'naive_box_regression'
    tracks = []
    mean_boxes = None

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        if idx == 1:
            scores, boxes = im_detect(net_rpn, im, mean_boxes)
        else:
            scores, boxes = im_detect(net_no_rpn, im, mean_boxes)
        mean_boxes = np.mean(boxes.reshape((boxes.shape[0], -1, 4)), axis=1)
        # mean_boxes = np.insert(mean_boxes, 0, 0, axis=1)
        _append_boxes(tracks, frame['frame'], mean_boxes, scores)
        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    track_proto['tracks'] = tracks
    return track_proto
