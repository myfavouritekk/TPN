#!/usr/bin/env python

import argparse
import os
import os.path as osp
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('window_file',
        help='Window file.')
    parser.add_argument('save_dir',
        help='Save directory.')
    parser.add_argument('--num_cls', type=int, default=31,
        help='Number of classes. [31]')
    parser.add_argument('--copy_file', action='store_true',
        help='Copy image file to the save_dir')
    parser.set_defaults(copy_file=False)
    args = parser.parse_args()

    results = dict([(idx, []) for idx in xrange(args.num_cls)])
    f = open(args.window_file)
    while f.readline(): # number line
        img_name = f.readline().strip()
        for _ in xrange(4): f.readline()
        num_rois = int(f.readline().strip())
        classes = []
        for _ in xrange(num_rois):
            classes.append(int(f.readline().strip().split()[0]))
        uniq_cls = set(classes)
        for cls in uniq_cls:
            if cls == 0: continue
            results[cls].append(img_name)
    f.close()

    for cls in results:
        if cls == 0: continue
        if args.copy_file:
            cls_dir = osp.join(args.save_dir, '{:02d}'.format(cls))
            if not osp.isdir(cls_dir): os.makedirs(cls_dir)
            for img_path in results[cls]:
                shutil.copyfile(img_path, cls_dir)
        else:
            with open(osp.join(args.save_dir,
                    '{:02d}.txt'.format(cls)), 'w') as f:
                for img_path in results[cls]:
                    f.write(img_path + '\n')
