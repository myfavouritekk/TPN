#!/usr/bin/env python

import init
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from rpn.generate import im_proposals
import numpy as np
import caffe, os, sys, cv2
import argparse
import cPickle

from vdetlib.utils.protocol import proto_load, proto_dump, frame_path_at
from vdetlib.vdet.dataset import imagenet_vdet_classes, index_vdet_to_det
from vdetlib.vdet.video_det import rpn_fast_rcnn_det_vid

NETS = {'vgg16': ('VGG16', 'rpn_test.pt',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'det': ('DET', 'bn_3k_pretrain_yangbin_proposals_900_crop8_dropout_5b_deploy.prototxt',
                  'hkbn_4d_fast_rcnn_iter_160000_post_process.caffemodel'),
        'vid': ('VID', 'test_4d_900_vid_crop8.prototxt',
                  'hkbn_4d_frcn_vid_iter_80000_post_processed.caffemodel')}
CLS_IDX = {'det': sorted(index_vdet_to_det.values()),
           'vid': sorted(index_vdet_to_det.keys())}

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('vid_file')
    parser.add_argument('save_file')
    parser.add_argument('--job', help='Job slot, GPU ID + 1. [1]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--rpn', dest='rpn', help='Network for RPN [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--fast_rcnn', dest='feature_net',
                        help='Network for fast r-cnn [det].',
                        choices=NETS.keys(), default='det')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # cfg.TEST.HAS_RPN = False  # Use RPN for proposals

    args = parse_args()
    model_dir = 'models'

    # rpn model
    rpn_pt = os.path.join(model_dir, args.rpn, NETS[args.rpn][1])
    rpn_model = os.path.join(model_dir, args.rpn, NETS[args.rpn][2])

    # fast_rcnn model
    fast_rcnn_pt = os.path.join(model_dir, args.feature_net, NETS[args.feature_net][1])
    fast_rcnn_model = os.path.join(model_dir, args.feature_net, NETS[args.feature_net][2])

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        gpu_id = args.job - 1;
        caffe.set_device(gpu_id)
        cfg.GPU_ID = gpu_id
    print '\n\nLoaded network {:s}'.format(rpn_model)
    rpn_net = caffe.Net(rpn_pt, rpn_model, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(fast_rcnn_model)
    fast_rcnn_net = caffe.Net(fast_rcnn_pt, fast_rcnn_model, caffe.TEST)

    vid_proto = proto_load(args.vid_file)

    gen_rois = lambda net, im: im_proposals(net, im)[0]
    det_proto = rpn_fast_rcnn_det_vid(rpn_net, fast_rcnn_net, vid_proto, gen_rois,
        im_detect, CLS_IDX[args.feature_net])

    save_dir = os.path.dirname(args.save_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    proto_dump(det_proto, args.save_file)
