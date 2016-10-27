#!/usr/bin/env python
import numpy as np
import numpy.random as npr
import yaml
import caffe
import scipy.io as sio
import cPickle
import cv2
import random
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from utils.blob import prep_im_for_blob, im_list_to_fixed_spatial_blob

class PairROIDataLayer(caffe.Layer):
    """docstring for PairROIDataLayer"""
    def setup(self, bottom, top):
        with open(self.param_str) as f:
            layer_params = yaml.load(f)
        self.config = layer_params
        # n image pairs
        with open(self.config['source']) as f:
            self.imagelist = [line.strip().split() for line in f]

        # n cells containing bounding boxes
        self.bbox = sio.loadmat(self.config['bbox'])['bbox']
        assert len(self.bbox) == len(self.imagelist)
        # n cells containing gt for both images
        self.gt = sio.loadmat(self.config['gt'])['gt']
        assert len(self.gt) == len(self.imagelist)

        self.index = range(len(self.imagelist))
        if self.config['shuffle']:
            random.shuffle(self.index)
        self.iter = 0

        with open(self.config['bbox_mean'], 'rb') as f:
            self.bbox_mean = cPickle.load(f)
        with open(self.config['bbox_std'], 'rb') as f:
            self.bbox_std = cPickle.load(f)

        # setup shapes
        # image pair
        top[0].reshape(2, 3, cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE)
        # rois (img_idx, x1, y1, x2, y2)
        top[1].reshape(self.config['batch_size'] * 2, 5)
        # labels
        top[2].reshape(self.config['batch_size'], 1)
        # bbox_targets
        top[3].reshape(self.config['batch_size'], 4)
        # bbox_weights
        top[4].reshape(self.config['batch_size'], 4)

    def forward(self, bottom, top):
        index = self.index[self.iter]
        print index
        raise
        img_name1, img_name2 = self.imagelist[index]
        img1, scale = _image_preprocess(cv2.imread(img_name1))
        img2, __ = _image_preprocess(cv2.imread(img_name2))
        blobs = im_list_to_fixed_spatial_blob([img1, img2],
            cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE)
        bboxes  = self.bbox[index]
        gt1, gt2 = self.gt[index]
        self.iter += 1
        if self.iter == len(self.imagelist):
            self.iter = 0

        # sample rois
        overlaps = bbox_overlaps(bboxes, gt1)
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)

        fg_inds = np.where(max_overlaps >= self.config['select_overlap'])[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = min(self.config['batch_size'], fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

        bg_inds = np.where(max_overlaps < self.config['select_overlap'])[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = self.config['batch_size'] - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions without replacement
        if bg_inds.size > 0:
            bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

        labels = np.ones((self.config['batch_size'], 1), dtype=np.float)
        labels[fg_rois_per_this_image:] = 0

        keep_ids = np.append(fg_inds, bg_inds)

        # n * 1 * 4
        rois = bboxes[keep_ids][:,np.newaxis,:]
        rois = np.tile(rois, (1, 2, 1))
        rois  = rois * scale # scale rois to match image scale
        assignment = np.zeros((self.config['batch_size'], 2, 1), dtype=np.float)
        assignment[:,1,:] = 1
        rois = np.concatenate((assignment, rois), axis=2).reshape((-1, 5))
        print rois[0,0]
        print rois[0,1]
        print rois[1,0]
        print rois[1,1]
        import pdb
        pdb.set_trace()

        # compute targets and weights
        bbox_targets = bbox_transform(gt1[gt_assignment[keep_ids]],
                                          gt2[gt_assignment[keep_ids]])
        bbox_targets = (bbox_targets - self.bbox_mean) / self.bbox_std
        bbox_weights = np.zeros_like(bbox_targets)
        bbox_weights[labels, ...] = 1

        top[0].data[...] = blobs
        top[1].data[...] = rois
        top[2].data[...] = labels
        top[3].data[...] = bbox_targets
        top[4].data[...] = bbox_weights

    def backward(self, bottom, top):
        pass

def _image_preprocess(img):
    target_scale = npr.choice(cfg.TRAIN.SCALES)
    im, im_scale = prep_im_for_blob(img, cfg.PIXEL_MEANS, target_scale,
                                    cfg.TRAIN.MAX_SIZE)
    return im, im_scale
