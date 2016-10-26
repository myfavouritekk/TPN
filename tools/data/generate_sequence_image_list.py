#!/usr/bin/env python

import argparse
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('orig_list')
    parser.add_argument('root_dir')
    parser.add_argument('save_list')
    parser.add_argument('--length', type=int, default=2)
    args = parser.parse_args()

    with open(args.orig_list) as f:
        image_list = [line.strip() for line in f]

    with open(args.save_list, 'w') as f:
        for img_path in image_list:
            count = int(img_path.split('/')[-1])
            vid_dir = '/'.join(img_path.split('/')[:-1])
            img1 = osp.join(args.root_dir, img_path+'.JPEG')
            assert osp.isfile(img1)
            img_paths = []
            all_exist = True
            for i in xrange(args.length):
                img2_path = "{}/{:06d}.JPEG".format(vid_dir, count+i)
                img2 = osp.join(args.root_dir, img2_path)
                if osp.isfile(img2):
                    img_paths.append(img2_path)
                else:
                    all_exist = False
                    break
            if all_exist:
                f.write('{}\n'.format(" ".join(img_paths)))
