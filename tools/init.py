#!/usr/bin/env python

"""Setup paths for TPN"""


import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
ext_dir = osp.join(this_dir, '..', 'external')

# Add py-faster-rcnn paths to PYTHONPATH
frcn_dir = osp.join(this_dir, '..', 'external', 'py-faster-rcnn')
add_path(osp.join(frcn_dir, 'lib'))
# caffe_path = osp.join('/Volumes/Research/ECCV2016/Code/External/fast-rcnn-VID-test', 'caffe-fast-rcnn', 'python')
add_path(osp.join(frcn_dir, 'caffe-fast-rcnn', 'python'))

# Add vdetlib to PYTHONPATH
lib_path = ext_dir
add_path(lib_path)

# tpn related modules
src_dir = osp.join(this_dir, '..', 'src')
add_path(src_dir)