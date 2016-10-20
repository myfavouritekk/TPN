#!/usr/bin/env python
import argparse
import scipy.io as sio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('proposal_file')
    parser.add_argument('select_list')
    parser.add_argument('save_file')
    args = parser.parse_args()

    with open(args.select_list) as f:
        selections = [line.strip().split()[0] for line in f]
    # remove .JPEG
    # selections = [img[:-5] for img in selections]
    mat = sio.loadmat(args.proposal_file)
    boxes = mat['boxes']
    imgs = mat['images']
    imgs = [img[0][0] for img in imgs]
    res = []
    for selection in selections:
        assert selection in imgs
        res.append(boxes[imgs.index(selection)])

    sio.savemat(args.save_file, {'bbox': res}, do_compression=True)
