#!/usr/bin/env python
# propagate bounding boxes

from fast_rcnn.test import im_detect
from fast_rcnn.bbox_transform import bbox_transform_inv
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


def _box_proto_to_track(box_proto, max_frame, length, sample_rate, offset=0):
    # generate empty tracks according to box proto
    tracks = []
    for box in box_proto['boxes']:
        if (box['frame'] - 1) % sample_rate != offset: continue
        track = []
        for i in xrange(length):
            if box['frame'] + i > max_frame: break
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

def _update_track(tracks, pred_boxes, scores, features, track_index, frame_id):
    if features is not None:
        for i, pred_box, cls_scores, feat in zip(track_index, pred_boxes, scores, features):
            for box in tracks[i]:
                if box['frame'] == frame_id:
                    box['bbox'] = pred_box.tolist()
                    box['scores'] = cls_scores.tolist()
                    box['feature'] = feat.tolist()
                if box['frame'] == frame_id + 1:
                    box['roi'] = pred_box.tolist()
                    break
    else:
        for i, pred_box, cls_scores in zip(track_index, pred_boxes, scores):
            for box in tracks[i]:
                if box['frame'] == frame_id:
                    box['bbox'] = pred_box.tolist()
                    box['scores'] = cls_scores.tolist()
                if box['frame'] == frame_id + 1:
                    box['roi'] = pred_box.tolist()
                    break
"""
- modified track_proto

    ```json
        {
            "video": "video_name",
            "method": "tracking_method_name",
            "tracks": [
                [
                    {
                        "frame": 1,
                        "bbox": [x1, y1, x2, y2], // bbox predictions
                        // regions of interest, bbox before prediction
                        "roi": [x1, y1, x2, y2],
                        "scores": [...],
                        "features": [....]
                        "anchor": int
                    },
                    {
                        "frame": 2,
                        "bbox": [x1, y1, x2, y2], // bbox predictions
                        // regions of interest, bbox before prediction
                        "roi": [x1, y1, x2, y2],
                        "scores": [...],
                        "features": [....]
                        "anchor": int
                    }
                ],  // tracklet 1
                [
                    // tracklet 2
                ]
                // ...
            ]
        }
    ```
"""

def roi_propagation(vid_proto, box_proto, net, det_fun=im_detect, scheme='max', length=None,
        sample_rate=1, offset=0, cls_indices=None, keep_feat=False):
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'roi_propagation'
    max_frame = vid_proto['frames'][-1]['frame']
    if not length: length = max_frame
    tracks = _box_proto_to_track(box_proto, max_frame, length, sample_rate,
                offset)

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois, track_index = _cur_rois(tracks, frame['frame'])
        if len(rois) == 0: continue
        # print "Frame {}: {} proposals".format(frame['frame'], len(rois))
        timer = Timer()
        timer.tic()


        # scores: n x c, boxes: n x (c x 4)
        scores = []
        boxes = []
        features = []
        # split to several batches to avoid memory error
        batch_size = 1024
        for roi_batch in np.split(np.asarray(rois), range(0, len(rois), batch_size)[1:]):
            s_batch, b_batch = det_fun(net, im, np.asarray(roi_batch))
            f_batch = net.blobs['global_pool'].data.copy().squeeze(axis=(2,3))
            scores.append(s_batch)
            boxes.append(b_batch)
            features.append(f_batch)
        scores = np.concatenate(scores, axis=0)
        boxes = np.concatenate(boxes, axis=0)
        boxes = boxes.reshape((boxes.shape[0], -1, 4))
        if keep_feat:
            features = np.concatenate(features, axis=0)
            assert features.shape[0] == scores.shape[0]
        else:
            features = None
        if cls_indices is not None:
            boxes = boxes[:, cls_indices, :]
            scores = scores[:, cls_indices]
            # scores normalization
            scores = scores / np.sum(scores, axis=1, keepdims=True)

        # propagation schemes
        if scheme == 'mean':
            # use mean regressions as predictios
            pred_boxes = np.mean(boxes, axis=1)
        elif scheme == 'max':
            # use the regressions of the class with the maximum probability
            # excluding __background__ class
            max_cls = scores[:,1:].argmax(axis=1) + 1
            pred_boxes = boxes[np.arange(len(boxes)), max_cls, :]
        elif scheme == 'weighted':
            # use class specific regression as predictions
            cls_boxes = boxes[:,1:,:]
            cls_scores = scores[:,1:]
            pred_boxes = np.sum(cls_boxes * cls_scores[:,:,np.newaxis], axis=1) / np.sum(cls_scores, axis=1, keepdims=True)
        else:
            raise ValueError("Unknown scheme {}.".format(scheme))

        # update track bbox
        _update_track(tracks, pred_boxes, scores, features, track_index, frame['frame'])
        timer.toc()
        print ('Frame {}: Detection took {:.3f}s for '
               '{:d} object proposals').format(frame['frame'], timer.total_time, len(rois))
    track_proto['tracks'] = tracks
    return track_proto


def tpn_test(vid_proto, box_proto, net, rnn_net, session, det_fun=im_detect, scheme='max', length=None,
        sample_rate=1, offset=0, cls_indices=None, batch_size=64):
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'roi_propagation'
    max_frame = vid_proto['frames'][-1]['frame']
    if not length: length = max_frame
    tracks = _box_proto_to_track(box_proto, max_frame, length, sample_rate,
                offset)

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois, track_index = _cur_rois(tracks, frame['frame'])
        if len(rois) == 0: continue
        # print "Frame {}: {} proposals".format(frame['frame'], len(rois))
        timer = Timer()
        timer.tic()

        # scores: n x c, boxes: n x (c x 4)
        scores = []
        boxes = []
        features = []
        # split to several batches to avoid memory error
        for roi_batch in np.split(np.asarray(rois), range(0, len(rois), batch_size)[1:]):
            num_rois = roi_batch.shape[0]
            roi_holder = np.zeros((batch_size, 4), dtype=np.float32)
            roi_holder[:num_rois,:] = np.asarray(roi_batch)
            s_batch, b_batch = det_fun(net, im, roi_holder)
            f_batch = net.blobs['global_pool'].data.copy().squeeze(axis=(2,3))
            scores.append(s_batch[:num_rois,...])
            boxes.append(b_batch[:num_rois,...])
            features.append(f_batch[:num_rois,...])
        scores = np.concatenate(scores, axis=0)
        boxes = np.concatenate(boxes, axis=0)
        boxes = boxes.reshape((boxes.shape[0], -1, 4))
        features = np.concatenate(features, axis=0)
        assert features.shape[0] == scores.shape[0]

        if cls_indices is not None:
            boxes = boxes[:, cls_indices, :]
            scores = scores[:, cls_indices]
            # scores normalization
            scores = scores / np.sum(scores, axis=1, keepdims=True)

        # propagation schemes
        if scheme == 'mean':
            # use mean regressions as predictios
            pred_boxes = np.mean(boxes, axis=1)
        elif scheme == 'max':
            # use the regressions of the class with the maximum probability
            # excluding __background__ class
            max_cls = scores[:,1:].argmax(axis=1) + 1
            pred_boxes = boxes[np.arange(len(boxes)), max_cls, :]
        elif scheme == 'weighted':
            # use class specific regression as predictions
            cls_boxes = boxes[:,1:,:]
            cls_scores = scores[:,1:]
            pred_boxes = np.sum(cls_boxes * cls_scores[:,:,np.newaxis], axis=1) / np.sum(cls_scores, axis=1, keepdims=True)
        else:
            raise ValueError("Unknown scheme {}.".format(scheme))

        # update track bbox
        _update_track(tracks, pred_boxes, scores, features, track_index, frame['frame'])
        timer.toc()
        print ('Frame {}: Detection took {:.3f}s for '
               '{:d} object proposals').format(frame['frame'], timer.total_time, len(rois))
    print 'Running LSTM...'
    for track in tracks:
        feat = np.asarray([box['feature'] for box in track])
        track_length = len(track)
        expend_feat = np.zeros((rnn_net.num_steps,) + feat.shape[1:])
        expend_feat[:track_length] = feat

        # extract features
        state = session.run([rnn_net.initial_state])
        cls_scores, bbox_deltas, end_probs, state = session.run(
            [rnn_net.cls_scores, rnn_net.bbox_pred, rnn_net.end_probs,
            rnn_net.final_state],
            {rnn_net.input_data: expend_feat[np.newaxis,:,:],
             rnn_net.initial_state: state[0]})

        # process outputs
        rois = np.asarray([box['roi'] for box in track])
        bbox_pred = bbox_transform_inv(rois, bbox_deltas[:track_length,:])
        cls_pred_lstm = cls_scores[:track_length]
        end_probs = end_probs[:track_length]
        for box, cur_bbox_pred, cur_cls_pred_lstm, cur_end_prob in \
            zip(track, bbox_pred, cls_pred_lstm, end_probs):
            box['scores_lstm'] = cur_cls_pred_lstm.tolist()
            box['bbox_lstm'] = cur_bbox_pred.tolist()
            box['end_prob'] = float(cur_end_prob)
            del box['feature']
    track_proto['tracks'] = tracks
    return track_proto
