#!/usr/bin/env python

import zipfile
import cPickle
import numpy as np
import glob
import os.path as osp
import random

"""
    track_obj: {
        frames: 1 by n numpy array,
        anchors: 1 by n numpy array,
        features: m by n numpy array,
        scores: c by n numpy array,
        boxes: 4 by n numpy array,
        rois: 4 by n numpy array
    }
"""
def save_track_proto_to_zip(track_proto, save_file):
    zf = zipfile.ZipFile(save_file, 'w', allowZip64=True)
    print "Writing to zip file {}...".format(save_file)
    for track_id, track in enumerate(track_proto['tracks']):
        track_obj = {}
        for key in track[0]:
            track_obj[key] = np.asarray([box[key] for box in track])
        zf.writestr('{:06d}.pkl'.format(track_id),
            cPickle.dumps(track_obj, cPickle.HIGHEST_PROTOCOL))
        if (track_id + 1) % 1000 == 0:
            print "\t{} tracks written.".format(track_id + 1)
    print "\tTotally {} tracks written.".format(track_id + 1)
    zf.close()


def tpn_raw_data(data_path):
    # return the paths of training and validation zip files
    train_set = sorted(glob.glob(osp.join(data_path, 'train/*')))
    val_set = sorted(glob.glob(osp.join(data_path, 'val/*')))
    valid_train_set = []
    valid_val_set = []
    for set_name, orig_set, valid_set in \
            [('train', train_set, valid_train_set), ('val', val_set, valid_val_set)]:
        print "Checking {} set files...".format(set_name)
        for ind, orig_vid in enumerate(orig_set, start=1):
            if zipfile.is_zipfile(orig_vid):
                valid_set.append(orig_vid)
            elif osp.isdir(orig_vid):
                valid_set.append(orig_vid)
            else:
                print "{} is not a valid zip file or a directory".format(orig_vid)
            if ind % 1000 == 0:
                print "{} files checked.".format(ind)
        if ind % 1000 != 0:
            print "Totally {} files checked.".format(ind)
    return valid_train_set, valid_val_set


def _expand_bbox_targets(bbox_targets, class_labels, num_classes, num_steps):
    # expend_targets: num_steps * (num_classes * 4)
    # weights: num_steps * (num_classes * 4)
    expend_targets = np.zeros((num_steps, num_classes, 4), dtype=np.float)
    weights = np.zeros_like(expend_targets, dtype=np.float)
    for ind, (target, cls) in enumerate(zip(bbox_targets, class_labels)):
        if cls == 0: continue
        expend_targets[ind, cls, :] = np.asarray(target).flatten()
        weights[ind, cls, :] = 1.
    return expend_targets.reshape((num_steps, -1)), weights.reshape((num_steps, -1))


def tpn_iterator(raw_data, batch_size, num_steps, num_classes, num_vids, fg_ratio=None):
    """ return values:
            x, cls_t, bbox_t, end_t
            x: input features
                [batch_size, num_steps, input_size]
            cls_t: classification targets
                [batch_size, num_steps]
            end_t: ending prediction targets
                [batch_size, num_steps]
            bbox_t: bounding box regression targets
                [batch_size, num_steps, num_classes * 4]
            bbox_weights: bounding box regression weights
                [batch_size, num_steps, num_classes * 4]
    """
    keys = ['feature', 'class_label', 'end_label', 'bbox_target']
    temp_res = {}
    for key in keys: temp_res[key] = None
    temp_res['bbox_weights'] = None
    rand_vids = random.sample(raw_data, num_vids)
    assert batch_size % num_vids == 0
    sample_per_vid = batch_size / num_vids

    for vid_ind, vid in enumerate(rand_vids):
        tracks = []
        # zipfile
        if zipfile.is_zipfile(vid):
            zf = zipfile.ZipFile(vid)
            track_list = zf.namelist()
            if fg_ratio is None:
                # natural distribution
                track_samples = sorted(random.sample(track_list, sample_per_vid))
            else:
                raise NotImplementedError('Track foreground ratio is not yet supported.')
            for track_name in track_samples:
                tracks.append(cPickle.loads(zf.read(track_name)))
            zf.close()
        # folders
        elif osp.isdir(vid):
            track_list = glob.glob(osp.join(vid, '*'))
            if fg_ratio is None:
                # natural distribution
                track_samples = sorted(random.sample(track_list, sample_per_vid))
            else:
                raise NotImplementedError('Track foreground ratio is not yet supported.')
            for track_name in track_samples:
                tracks.append(cPickle.loads(open(track_name, 'rb').read()))
        else:
            raise NotImplementedError('Only zipfile and directories are supported.')

        # process track data
        for ind, track in enumerate(tracks):
            offset = vid_ind * sample_per_vid + ind
            for key in keys:
                if key == 'bbox_target':
                    targets, weights = _expand_bbox_targets(track[key],
                        track['class_label'], num_classes, num_steps)
                    if temp_res[key] is None:
                        # initialize temp_res[key]
                        temp_res[key] = np.zeros((batch_size,)+targets.shape,
                            dtype=targets.dtype)
                    temp_res[key][offset,...] = targets
                    if temp_res['bbox_weights'] is None:
                        temp_res['bbox_weights'] = np.zeros((batch_size,)+weights.shape,
                            dtype=weights.dtype)
                    temp_res['bbox_weights'][offset,...] = weights
                else:
                    track_length = track[key].shape[0]
                    if key == 'class_label':
                        expend_res = -np.ones((num_steps,) + track[key].shape[1:])
                    elif key == 'end_label':
                        expend_res = np.ones((num_steps,) + track[key].shape[1:])
                    else:
                        expend_res = np.zeros((num_steps,) + track[key].shape[1:])
                    expend_res[:track_length] = track[key]
                    if temp_res[key] is None:
                        # initialize temp_res[key]
                        temp_res[key] = np.zeros((batch_size,)+expend_res.shape,
                            dtype=expend_res.dtype)
                    temp_res[key][offset, ...] = expend_res
    # collect all results
    res = []
    for key in keys:
        res.append(temp_res[key])
    res.append(temp_res['bbox_weights'])
    return tuple(res)


def tpn_test_iterator(track_path):
    """ return values:
            x: list of tracks
    """
    temp_res = None

    tracks = []
    # zipfile
    if zipfile.is_zipfile(track_path):
        zf = zipfile.ZipFile(track_path)
        track_list = zf.namelist()
        # print "Loading {} tracks...".format(len(track_list))
        for track_name in track_list:
            tracks.append(cPickle.loads(zf.read(track_name)))
        zf.close()
    # folders
    elif osp.isdir(track_path):
        track_list = sorted(glob.glob(osp.join(track_path, '*')))
        # print "Loading {} tracks...".format(len(track_list))
        for track_name in track_list:
            tracks.append(cPickle.loads(open(track_name, 'rb').read()))
    else:
        raise NotImplementedError('Only zipfile and directories are supported.')

    return tracks
