#!/usr/bin/env python

import argparse
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('orig_list')
    parser.add_argument('root_dir')
    parser.add_argument('save_list')
    args = parser.parse_args()

    with open(args.orig_list) as f:
        image_list = [line.strip() for line in f]

    with open(args.save_list, 'w') as f:
        for img_path in image_list:
            count = int(img_path.split('/')[-1])
            vid_dir = '/'.join(img_path.split('/')[:-1])
            img1 = osp.join(args.root_dir, img_path+'.JPEG')
            assert osp.isfile(img1)
            img2_path = "{}/{:06d}.JPEG".format(vid_dir, count+1)
            img2 = osp.join(args.root_dir, img2_path)
            if osp.isfile(img2):
                f.write('{} {}\n'.format(img_path+'.JPEG', img2_path))
