#!/usr/bin/env python
import argparse
import scipy.io as sio
import xmltodict
import os.path as osp
import numpy as np
from functools import reduce

def load_gt(xml_file):
    res = []
    with open(xml_file) as f:
        xml = dict(xmltodict.parse(f.read())['annotation'])
        try:
            obj = xml['object']
        except KeyError:
            print "xml {} has no objects.".format(xml_file)
            return np.asarray(res)
        if type(obj) is not list:
            boxes = [obj]
        else:
            boxes = obj
        for box in boxes:
            track_id = str(box['trackid'])
            bbox = map(int, [box['bndbox']['xmin'],
                             box['bndbox']['ymin'],
                             box['bndbox']['xmax'],
                             box['bndbox']['ymax'],
                             track_id])
            res.append(bbox)
    return np.asarray(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_list')
    parser.add_argument('gt_root')
    parser.add_argument('save_file')
    args = parser.parse_args()

    with open(args.img_list) as f:
        img_list = [line.strip().split() for line in f]
    res = []
    for img_paths in img_list:
        track_ids = []
        gts = []
        for img_path in img_paths:
            gt = load_gt(osp.join(args.gt_root, img_path[:-5] + '.xml'))
            gts.append(gt)
            try:
                track_ids.append(gt[:,-1])
            except IndexError:
                track_ids.append([])
        select_track_id = reduce(np.intersect1d, track_ids)
        cur_res = []
        assert len(gts) == len(track_ids)
        for track_id, gt in zip(track_ids, gts):
            cur_indices = [track_id.tolist().index(track_idx) for track_idx in select_track_id]
            if cur_indices:
                cur_res.append(gt[cur_indices,:4])
            else:
                cur_res.append([])
        res.append(cur_res)
    sio.savemat(args.save_file, {'gt': res}, do_compression=True)

