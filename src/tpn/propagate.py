#!/usr/bin/env python
# propagate bounding boxes

from fast_rcnn.craft import im_detect, sequence_im_detect
from fast_rcnn.bbox_transform import bbox_transform_inv, bbox_transform
from utils.cython_bbox import bbox_overlaps
from vdetlib.utils.protocol import frame_path_at, boxes_at_frame
from vdetlib.utils.timer import Timer
from vdetlib.utils.common import imread
import numpy as np
import random
import copy
import math
import itertools

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
                try:
                    rois.append(box['roi'])
                except:
                    import pdb
                    pdb.set_trace()
                index.append(track_id)
                break
    return rois, index

def _update_track(tracks, cls_boxes, pred_boxes, scores, features, track_index, frame_id):
    if features is not None:
        for i, cls_bbox, pred_box, cls_scores, feat in zip(track_index,
                cls_boxes, pred_boxes, scores, features):
            for box in tracks[i]:
                if box['frame'] == frame_id:
                    box['bbox'] = cls_bbox.tolist()
                    box['scores'] = cls_scores.tolist()
                    box['feature'] = feat.tolist()
                if box['frame'] == frame_id + 1:
                    box['roi'] = pred_box.tolist()
                    break
    else:
        for i, cls_bbox, pred_box, cls_scores in zip(track_index,
                cls_boxes, pred_boxes, scores):
            for box in tracks[i]:
                if box['frame'] == frame_id:
                    box['bbox'] = cls_bbox.tolist()
                    box['scores'] = cls_scores.tolist()
                if box['frame'] == frame_id + 1:
                    box['roi'] = pred_box.tolist()
                    break

def _update_track_by_key(tracks, key, values, track_index, frame_id):
    assert len(values) == len(track_index)
    for i, single_value in zip(track_index, values):
        for box in tracks[i]:
            if box['frame'] == frame_id:
                box[key] = single_value
                break

def _update_track_scores_boxes(tracks, scores, boxes, features, track_index, frame_id):
    if features is not None:
        for i, cls_scores, bbox, feat in zip(track_index, scores, boxes, features):
            for box in tracks[i]:
                if box['frame'] == frame_id:
                    box['scores'] = cls_scores.tolist()
                    box['bbox'] = bbox.tolist()
                    box['feature'] = feat.tolist()
                    break
    else:
        for i, cls_scores, bbox in zip(track_index, scores, boxes):
            for box in tracks[i]:
                if box['frame'] == frame_id:
                    box['scores'] = cls_scores.tolist()
                    box['bbox'] = bbox.tolist()
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

def _batch_im_detect(net, im, rois, det_fun, batch_size):
    # scores: n x c, boxes: n x (c x 4)
    scores = []
    boxes = []
    features = []
    # split to several batches to avoid memory error
    rois = np.asarray(rois)
    num_rois = len(rois)
    num_batches = int(np.ceil(float(num_rois) / batch_size))
    rois_holder = np.tile(rois[0, :], (num_batches*batch_size, 1))
    rois_holder[:num_rois, :] = rois
    for j in xrange(num_batches):
        roi_batch = rois_holder[j*batch_size:(j+1)*batch_size, :]
        s_batch, b_batch = det_fun(net, im, roi_batch)
        f_batch = net.blobs['global_pool'].data.copy().squeeze(axis=(2,3))
        # must copy() because of batches may overwrite each other
        scores.append(s_batch.copy())
        boxes.append(b_batch.copy())
        features.append(f_batch.copy())
    scores = np.vstack(scores)[:num_rois]
    boxes = np.vstack(boxes)[:num_rois]
    boxes = boxes.reshape((boxes.shape[0], -1, 4))[:num_rois]
    features = np.vstack(features)[:num_rois]
    return scores.copy(), boxes.copy(), features.copy()

def score_guided_box_merge(scores, boxes, scheme):
    if scheme == 'mean':
        # use mean regressions as predictios
        return np.mean(boxes, axis=1)
    elif scheme == 'max':
        # use the regressions of the class with the maximum probability
        # excluding __background__ class
        max_cls = scores[:,1:].argmax(axis=1) + 1
        return boxes[np.arange(len(boxes)), max_cls, :].copy()
    elif scheme == 'weighted':
        # use class specific regression as predictions
        cls_boxes = boxes[:,1:,:]
        cls_scores = scores[:,1:]
        return np.sum(cls_boxes * cls_scores[:,:,np.newaxis], axis=1) / np.sum(cls_scores, axis=1, keepdims=True)
    else:
        raise ValueError("Unknown scheme {}.".format(scheme))

def roi_propagation(vid_proto, box_proto, net, det_fun=im_detect, scheme='max', length=None,
        sample_rate=1, offset=0, cls_indices=None, keep_feat=False,
        batch_size = 1024):
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'roi_propagation'
    max_frame = vid_proto['frames'][-1]['frame']
    if not length: length = max_frame
    tracks = _box_proto_to_track(box_proto, max_frame, length, sample_rate, offset)

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois, track_index = _cur_rois(tracks, frame['frame'])
        if len(rois) == 0: continue

        timer = Timer()
        timer.tic()

        # scores: n x c, boxes: n x (c x 4)
        scores, boxes, features = _batch_im_detect(net, im, rois,
                                                   det_fun, batch_size)

        if not keep_feat:
            features = None
        if cls_indices is not None:
            boxes = boxes[:, cls_indices, :]
            scores = scores[:, cls_indices]
            # scores normalization
            scores = scores / np.sum(scores, axis=1, keepdims=True)

        # propagation schemes
        pred_boxes = score_guided_box_merge(scores, boxes, scheme)

        # update track bbox
        _update_track(tracks, boxes, pred_boxes, scores, features, track_index, frame['frame'])
        timer.toc()
        print ('Frame {}: Detection took {:.3f}s for '
               '{:d} object proposals').format(frame['frame'], timer.total_time, len(rois))
    track_proto['tracks'] = tracks
    return track_proto

def track_propagation(vid_proto, track_proto, net, det_fun=im_detect,
        cls_indices=None, keep_feat=False, batch_size = 1024):
    new_track_proto = {}
    new_track_proto['video'] = vid_proto['video']
    new_track_proto['method'] = 'track_propagation'
    tracks = copy.copy(track_proto['tracks'])

    for idx, frame in enumerate(vid_proto['frames'], start=1):
        # Load the demo image
        image_name = frame_path_at(vid_proto, frame['frame'])
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois, track_index = _cur_rois(tracks, frame['frame'])
        if len(rois) == 0: continue

        timer = Timer()
        timer.tic()

        # scores: n x c, boxes: n x (c x 4)
        scores, boxes, features = _batch_im_detect(net, im, rois,
                                                   det_fun, batch_size)

        if not keep_feat:
            features = None
        if cls_indices is not None:
            scores = scores[:, cls_indices]
            # scores normalization
            scores = scores / np.sum(scores, axis=1, keepdims=True)

        # update track scores and boxes
        _update_track_scores_boxes(tracks, scores, boxes, features,
            track_index, frame['frame'])
        timer.toc()
        print ('Frame {}: Detection took {:.3f}s for '
               '{:d} object proposals').format(frame['frame'], timer.total_time, len(rois))
    new_track_proto['tracks'] = tracks
    return new_track_proto

def tpn_test(vid_proto, box_proto, net, rnn_net, session, det_fun=im_detect, scheme='max', length=None,
        sample_rate=1, offset=0, cls_indices=None, batch_size=64):
    # same as roi_propagation except keep_feat is always True
    track_proto = roi_propagation(vid_proto, box_proto, net, det_fun=det_fun,
        scheme=scheme, length=length, sample_rate=sample_rate,
        offset=offset, cls_indices=cls_indices, batch_size=batch_size,
        keep_feat=True)

    print 'Running LSTM...'
    for track in track_proto['tracks']:
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
    return track_proto

def tpn_caffe_test(vid_proto, box_proto, net, rnn_net, det_fun=im_detect,
        scheme='weighted', length=None,
        sample_rate=1, offset=0, cls_indices=None, batch_size=64):
    # same as roi_propagation except keep_feat is always True
    track_proto = roi_propagation(vid_proto, box_proto, net, det_fun=det_fun,
        scheme=scheme, length=length, sample_rate=sample_rate,
        offset=offset, cls_indices=cls_indices, batch_size=batch_size,
        keep_feat=True)

    print 'Running LSTM...'
    cont = np.ones((length,1))
    cont[0,:] = 0
    rnn_net.blobs['cont'].reshape(*cont.shape)
    rnn_net.blobs['cont'].data[...] = cont

    for track_idx, track in enumerate(track_proto['tracks'], start=1):
        feat = np.asarray([box['feature'] for box in track])
        track_length = len(track)
        expend_feat = np.zeros((length, 1) + feat.shape[1:])
        expend_feat[:track_length,0] = feat

        # extract features
        rnn_net.blobs['data'].reshape(*expend_feat.shape)
        rnn_net.blobs['data'].data[...] = expend_feat
        blobs_out = rnn_net.forward()
        bbox_deltas = blobs_out['bbox_pred']
        cls_scores = blobs_out['cls_prob']
        if 'end_prob' in blobs_out:
            end_probs = blobs_out['end_prob']
        else:
            end_probs = np.zeros((length,1))

        # process outputs
        rois = np.asarray([box['roi'] for box in track])
        bbox_pred = bbox_transform_inv(rois, bbox_deltas[:track_length,0,:])
        cls_pred_lstm = cls_scores[:track_length]
        end_probs = end_probs[:track_length]
        for box, cur_bbox_pred, cur_cls_pred_lstm, cur_end_prob in \
            zip(track, bbox_pred, cls_pred_lstm, end_probs):
            box['scores_lstm'] = cur_cls_pred_lstm.flatten().tolist()
            box['bbox_lstm'] = cur_bbox_pred.tolist()
            box['end_prob'] = float(cur_end_prob)
            del box['feature']
        if track_idx % 500 == 0:
            print "{} tracks processed.".format(track_idx)
    if track_idx % 500 != 0:
            print "{} tracks processed.".format(track_idx)
    return track_proto


def _sample_boxes(boxes, tot_num, fg_ratio):
    if fg_ratio == None:
        return random.sample(boxes, tot_num)
    else:
        cur_boxes = boxes
        pos_boxes = [box for box in cur_boxes if box['positive']]
        neg_boxes = [box for box in cur_boxes if not box['positive']]
        num_pos = int(fg_ratio * tot_num)
        if len(pos_boxes) < num_pos:
            num_pos = len(pos_boxes)
        num_neg = tot_num - num_pos
        return random.sample(pos_boxes, num_pos) + random.sample(neg_boxes, num_neg)


def roi_train_propagation(vid_proto, box_proto, net, det_fun=im_detect,
        cls_indices=None, scheme='weighted',
        num_tracks=16, length=20, fg_ratio=None, batch_size=16):
    assert vid_proto['video'] == box_proto['video']
    # calculate the number of boxes on each frame
    all_boxes = {}
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        boxes = boxes_at_frame(box_proto, frame_id)
        if len(boxes) >= num_tracks: all_boxes[frame_id] = boxes

    try:
        st_frame = random.choice(all_boxes.keys())
    except:
        raise ValueError('{} has not valid frames for tracking.'.format(vid_proto['video']))
    st_boxes = _sample_boxes(all_boxes[st_frame], num_tracks, fg_ratio)

    results = [{'frame': -1} for i in xrange(length)]
    anchor = 0
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        if frame_id < st_frame: continue
        if anchor >= length: break

        res = results[anchor]
        res['frame'] = frame_id
        if anchor == 0: res['roi'] = np.asarray([st_box['bbox'] for st_box in st_boxes])

        # Load the demo image
        image_name = frame_path_at(vid_proto, frame_id)
        im = imread(image_name)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois = res['roi']
        assert rois.shape[0] == num_tracks

        timer = Timer()
        timer.tic()

        # scores: n x c, boxes: n x (c x 4), features: n * c
        scores, boxes, features = _batch_im_detect(net, im, rois,
                                                   det_fun, batch_size)

        if cls_indices is not None:
            boxes = boxes[:, cls_indices, :]
            scores = scores[:, cls_indices]
            # scores normalization
            scores = scores / np.sum(scores, axis=1, keepdims=True)

        # propagation schemes
        pred_boxes = score_guided_box_merge(scores, boxes, scheme)

        results[anchor]['bbox'] = boxes
        results[anchor]['feat'] = features
        if anchor+1 < length:
            results[anchor+1]['roi'] = pred_boxes
        anchor += 1
    return results

def _batch_sequence_im_detect(net, imgs, rois, det_fun, batch_size):
    # scores: n x c, boxes: n x (c x 4)
    scores = []
    boxes = []
    features = []
    # split to several batches to avoid memory error
    rois = np.asarray(rois)
    num_rois = len(rois)
    num_batches = int(np.ceil(float(num_rois) / batch_size))
    rois_holder = np.tile(rois[0, :], (num_batches*batch_size, 1))
    rois_holder[:num_rois, :] = rois
    for j in xrange(num_batches):
        roi_batch = rois_holder[j*batch_size:(j+1)*batch_size, :]
        s_batch, b_batch = det_fun(net, imgs, roi_batch)
        f_batch = net.blobs['global_pool'].data.copy().squeeze(axis=(2,3))
        # must copy() because of batches may overwrite each other
        scores.append(s_batch.copy())
        boxes.append(b_batch.copy())
        features.append(f_batch.copy())
    scores = np.vstack(scores)[:num_rois]
    boxes = np.vstack(boxes)[:num_rois]
    features = np.vstack(features)[:num_rois]
    return scores.copy(), boxes.copy(), features.copy()

def _sequence_frames(vid_proto, window, track_anchors, length):
    # index is frame_id - 1
    anchor_idx = [anchor - 1 for anchor in track_anchors]
    vid_len = len(vid_proto['frames'])
    n_per_track = int(math.ceil((length - 1.) / (window - 1)))
    seq_frames = []
    frames = copy.copy(vid_proto['frames'])
    frames += window * [frames[-1]]
    step = window - 1
    for st_idx in anchor_idx:
        assert frames[st_idx]['frame'] in track_anchors
        for i in xrange(n_per_track):
            cur_st = st_idx+i*step
            cur_ed = st_idx+i*step+window
            if cur_st >= vid_len:
                break
            seq_frames.append(frames[cur_st:cur_ed])
    return seq_frames

def sequence_roi_propagation(vid_proto, box_proto, net, det_fun=sequence_im_detect,
        window=2,
        scheme='max', length=None,
        sample_rate=1, offset=0, keep_feat=False,
        batch_size = 1024):
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'sequence_roi_propagation'
    max_frame = vid_proto['frames'][-1]['frame']
    if not length: length = max_frame
    tracks = _box_proto_to_track(box_proto, max_frame, length, sample_rate, offset)

    track_anchors = sorted(set([track[0]['frame'] for track in tracks]))
    sequence_frames = _sequence_frames(vid_proto, window,
        track_anchors, length)
    for idx, frames in enumerate(sequence_frames, start=1):
        # Load the demo image
        images = map(lambda x: imread(frame_path_at(vid_proto, x['frame'])), frames)

        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        rois, track_index = _cur_rois(tracks, frames[0]['frame'])
        if len(rois) == 0: continue

        timer = Timer()
        timer.tic()

        # scores: n x 2, boxes: n x ((len-1) x 4), features: n x (len x f)
        scores, boxes, features = _batch_sequence_im_detect(
            net, images, rois, det_fun, batch_size)

        if not keep_feat:
            features = None

        # update track bbox
        boxes = boxes.reshape((len(rois), len(images)-1, 4))
        if keep_feat:
            features = features.reshape((len(rois), len(images), -1))
        frame_ids = [frame['frame'] for frame in frames]
        prev_id = -1
        for i in xrange(len(images)):
            frame_id = frames[i]['frame']
            # stop when encounting duplicate frames
            if frame_id == prev_id:
                break
            prev_id = frame_id
            if i == 0:
                _update_track_by_key(tracks, 'bbox', rois, track_index, frame_id)
            else:
                # minus 1 because boxes[0] correspond to the second frame
                _update_track_by_key(tracks, 'bbox', boxes[:,i-1,:].tolist(), track_index, frame_id)
                _update_track_by_key(tracks, 'roi', boxes[:,i-1,:].tolist(), track_index, frame_id)
            if keep_feat:
                _update_track_by_key(tracks, 'feature', features[:,i,:].tolist(), track_index, frame_id)
        timer.toc()
        print ('Frame {}-{}: Detection took {:.3f}s for '
               '{:d} object proposals').format(frame_ids[0], frame_ids[-1], timer.total_time, len(rois))
    track_proto['tracks'] = tracks
    return track_proto

def _gt_propagate_boxes(boxes, annot_proto, frame_id, window, overlap_thres):
    pred_boxes = []
    annots = []
    for annot in annot_proto['annotations']:
        for idx, box in enumerate(annot['track']):
            if box['frame'] == frame_id:
                gt1 = box['bbox']
                deltas = []
                deltas.append(gt1)
                for offset in xrange(1, window):
                    try:
                        gt2 = annot['track'][idx+offset]['bbox']
                    except IndexError:
                        gt2 = gt1
                    delta = bbox_transform(np.asarray([gt1]), np.asarray([gt2]))
                    deltas.append(delta)
                annots.append(deltas)
    gt1s = [annot[0] for annot in annots]
    if not gt1s:
        # no grount-truth, boxes remain still
        return np.tile(np.asarray(boxes)[:,np.newaxis,:], [1,window-1,1])
    overlaps = bbox_overlaps(np.require(boxes, dtype=np.float),
                             np.require(gt1s, dtype=np.float))
    assert len(overlaps) == len(boxes)
    for gt_overlaps, box in zip(overlaps, boxes):
        max_overlap = np.max(gt_overlaps)
        max_gt = np.argmax(gt_overlaps)
        sequence_box = []
        if max_overlap < overlap_thres:
            for offset in xrange(1, window):
                sequence_box.append(box)
        else:
            for offset in xrange(1, window):
                delta = annots[max_gt][offset]
                sequence_box.append(
                    bbox_transform_inv(np.asarray([box]), delta)[0].tolist())
        pred_boxes.append((sequence_box))
    return np.asarray(pred_boxes)

def gt_motion_propagation(vid_proto, box_proto, annot_proto,
        window=2, length=None,
        sample_rate=1, offset=0, overlap_thres=0.5):
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = 'gt_motion_propagation'
    max_frame = vid_proto['frames'][-1]['frame']
    if not length: length = max_frame
    tracks = _box_proto_to_track(box_proto, max_frame, length, sample_rate, offset)

    track_anchors = sorted(set([track[0]['frame'] for track in tracks]))
    sequence_frames = _sequence_frames(vid_proto, window,
        track_anchors, length)
    for idx, frames in enumerate(sequence_frames, start=1):
        # Detect all object classes and regress object bounds
        # extract rois on the current frame
        try:
            rois, track_index = _cur_rois(tracks, frames[0]['frame'])
        except:
            import pdb
            pdb.set_trace()
        if len(rois) == 0: continue

        timer = Timer()
        timer.tic()

        boxes = _gt_propagate_boxes(rois, annot_proto,
            frames[0]['frame'], window, overlap_thres)
        features = None

        # update track bbox
        boxes = boxes.reshape((len(rois), window-1, 4))
        frame_ids = [frame['frame'] for frame in frames]
        prev_id = -1
        for i in xrange(window):
            frame_id = frames[i]['frame']
            # stop when encounting duplicate frames
            if frame_id == prev_id:
                break
            prev_id = frame_id
            if i == 0:
                _update_track_by_key(tracks, 'bbox', rois, track_index, frame_id)
            else:
                # minus 1 because boxes[0] correspond to the second frame
                _update_track_by_key(tracks, 'bbox', boxes[:,i-1,:].tolist(), track_index, frame_id)
                _update_track_by_key(tracks, 'roi', boxes[:,i-1,:].tolist(), track_index, frame_id)

        timer.toc()
        print ('Frame {}-{}: Detection took {:.3f}s for '
               '{:d} object proposals').format(frame_ids[0], frame_ids[-1], timer.total_time, len(rois))
    track_proto['tracks'] = tracks
    return track_proto
