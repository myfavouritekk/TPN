#!/usr/bin/env python
import argparse
import scipy.io as sio
import os
import os.path as osp
import numpy as np
from vdetlib.vdet.dataset import index_det_to_vdet

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert a window file for DET for VID.')
    parser.add_argument('window_file')
    parser.add_argument('save_window_file')
    args = parser.parse_args()

    f = open(args.window_file, 'r')
    save_file = open(args.save_window_file, 'w')
    boxes = []
    image_ind = 0
    count = 0
    while 1:
        image_ind += 1
        if image_ind % 1000 == 0:
            print "Processed {} files.".format(image_ind)
        # read number line
        number_line = f.readline().strip()
        if len(number_line) == 0: break # end of the file
        assert number_line[0] == '#'

        # read image line
        img_path = f.readline().strip()
        image_specs = []
        for i in xrange(4): image_specs.append(f.readline().strip())
        num = int(f.readline().strip())
        cur_boxes = []
        only_bg = True
        for i in xrange(num):
            box_target = map(float, f.readline().strip().split())
            # skip background or other non-vid classes
            if int(box_target[0]) not in index_det_to_vdet: continue

            # map DET index to VID
            box_target[0] = index_det_to_vdet[box_target[0]]
            cur_boxes.append(box_target)
            if box_target[0] != 0:
                only_bg = False
        if len(cur_boxes) == 0 or only_bg: continue

        save_file.write('# {}\n'.format(count))
        count += 1
        save_file.write('{}\n'.format(img_path))
        for i in xrange(4): save_file.write('{}\n'.format(image_specs[i]))
        selected_num = len(cur_boxes)
        save_file.write('{}\n'.format(selected_num))
        for box_target in cur_boxes:
            save_file.write('{:.0f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:f} {:f} {:f} {:f}\n'.format(*box_target))

    if image_ind % 1000 != 0:
        print "Processed {} files.".format(image_ind)

    f.close()
    save_file.close()
