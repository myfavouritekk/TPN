#!/usr/bin/env python
import argparse
import scipy.io as sio
import xmltodict
import os.path as osp
import numpy as np

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
    for img1, img2 in img_list:
        gt1 = load_gt(osp.join(args.gt_root, img1[:-5] + '.xml'))
        gt2 = load_gt(osp.join(args.gt_root, img2[:-5] + '.xml'))
        try:
            track_id1 = gt1[:,-1]
            track_id2 = gt2[:,-1]
        except IndexError:
            res.append([[],[]])
            continue
        select_track_id = np.intersect1d(track_id1, track_id2)
        indx1 = [track_id1.tolist().index(track_idx) for track_idx in select_track_id]
        indx2 = [track_id2.tolist().index(track_idx) for track_idx in select_track_id]
        res.append([gt1[indx1,:4], gt2[indx2,:4]])
    sio.savemat(args.save_file, {'gt': res}, do_compression=True)

