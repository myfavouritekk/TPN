#!/usr/bin/env python

import zipfile
import cPickle
import numpy as np
import glob
import os.path as osp

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
            else:
                print "{} is not a valid zip file".format(orig_vid)
            if ind % 1000 == 0:
                print "{} files checked.".format(ind)
        if ind % 1000 != 0:
            print "Totally {} files checked.".format(ind)
    return valid_train_set, valid_val_set

def tpn_iterator(raw_data, batch_size, num_steps, num_classes):
    """ return values:
            x, cls_t, bbox_t, end_t
            x: input features
                [batch_size, input_size, num_steps]
            cls_t: classification targets
                [batch_size, num_steps]
            bbox_t: bounding box regression targets
                [batch_size, num_steps, num_classes * 4]
            end_t: ending prediction targets
                [batch_size, num_steps]
    """
    pass
