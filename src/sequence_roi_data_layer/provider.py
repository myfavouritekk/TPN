#!/usr/bin/env python
import os.path as osp
import numpy as np
import numpy.random as npr
import yaml
import scipy.io as sio
import cPickle
import cv2
import random
from fast_rcnn.config import cfg
from fast_rcnn.craft import _get_image_blob
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from utils.blob import prep_im_for_blob, im_list_to_fixed_spatial_blob

class SequenceROIDataProvider():
    """docstring for SequenceROIDataLayer"""
    def __init__(self, param_str):
        self.param_str = param_str

        with open(self.param_str) as f:
            layer_params = yaml.load(f)
        self.config = layer_params
        # n image pairs
        with open(self.config['source']) as f:
            self.imagelist = [line.strip().split() for line in f]
        self.length = len(self.imagelist[0])
        self.root_dir = self.config['root']

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

    def forward(self, step = 1):
        selected = False
        while not selected:
            index = self.index[self.iter]
            img_names = self.imagelist[index]
            proc_imgs = []
            for img_name in img_names:
                img_path = osp.join(self.root_dir, img_name)
                assert osp.isfile(img_path)
                proc_img, scale = _get_image_blob(cv2.imread(img_path))
                proc_imgs.append(proc_img)
            blobs = np.vstack(proc_imgs)
            bboxes  = self.bbox[index][0][:,:4]
            gts = self.gt[index]
            self.iter += step
            if self.iter >= len(self.imagelist):
                self.iter -= len(self.imagelist)
            if gts[0].shape[0] > 0: selected = True

        # sample rois
        overlaps = bbox_overlaps(np.require(bboxes, dtype=np.float),
                                 np.require(gts[0], dtype=np.float))
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
        rois = np.tile(rois, (1, self.length, 1))
        rois  = rois * scale # scale rois to match image scale
        assignment = np.tile(np.arange(self.length), (self.config['batch_size'], 1))[:,:,np.newaxis]
        rois = np.concatenate((assignment, rois), axis=2).reshape((-1, 5))

        # compute targets and weights
        bbox_targets = []
        bbox_weights = []
        for gt in gts[1:]:
            cur_bbox_targets = bbox_transform(gts[0][gt_assignment[keep_ids]],
                                              gt[gt_assignment[keep_ids]])
            cur_bbox_weights = np.zeros_like(cur_bbox_targets)
            cur_bbox_weights[labels.flatten().astype('bool'), ...] = 1
            bbox_targets.append(cur_bbox_targets)
            bbox_weights.append(cur_bbox_weights)
        bbox_targets = np.hstack(bbox_targets)
        bbox_weights = np.hstack(bbox_weights)
        bbox_targets = (bbox_targets - self.bbox_mean) / self.bbox_std

        return blobs, rois, labels, bbox_targets, bbox_weights

