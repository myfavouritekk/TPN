#!/usr/bin/env python
import numpy as np

def write_ilsvrc_results_file(all_boxes, f, thres=0.01):
    num_images = len(all_boxes[0])
    num_classes = len(all_boxes)
    for im_ind in xrange(num_images):
        for cls_ind in xrange(num_classes):
            if cls_ind == 0:
                continue
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep_inds = np.where(dets[:, -1]>=thres)[0]
            dets = dets[keep_inds, :]
            # the VOCdevkit expects 1-based indices
            for k in xrange(dets.shape[0]):
                f.write('{:d} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                        format(im_ind+1, cls_ind, dets[k, -1],
                               dets[k, 0] + 1, dets[k, 1] + 1,
                               dets[k, 2] + 1, dets[k, 3] + 1))
