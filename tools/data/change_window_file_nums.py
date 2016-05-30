#!/usr/bin/env python
import argparse
import scipy.io as sio
import os
import os.path as osp
import numpy as np
from vdetlib.vdet.dataset import index_det_to_vdet

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Change window file numbers.')
    parser.add_argument('window_file')
    parser.add_argument('start', type=int)
    parser.add_argument('save_window_file')
    args = parser.parse_args()

    f = open(args.window_file, 'r')
    save_file = open(args.save_window_file, 'w')
    boxes = []
    image_ind = 0
    count = 0
    while 1:
        # read number line
        number_line = f.readline().strip()
        if len(number_line) == 0: break # end of the file
        assert number_line[0] == '#'

        save_file.write('# {}\n'.format(image_ind + args.start))
        # read image line and image specs
        for __ in xrange(5):
            save_file.write(f.readline())

        num = int(f.readline().strip())
        save_file.write('{}\n'.format(num))
        for i in xrange(num):
            save_file.write(f.readline())

        image_ind += 1
        if image_ind % 1000 == 0:
            print "Processed {} files.".format(image_ind)

    if image_ind % 1000 != 0:
        print "Processed {} files.".format(image_ind)

    f.close()
    save_file.close()
