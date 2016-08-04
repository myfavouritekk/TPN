#!/usr/bin/env python

import os
import os.path as osp
import numpy as np
import tensorflow as tf
from model import TPNModel
import argparse
import glog as log
import glob
from data_io import tpn_test_iterator
import cPickle

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
logging = tf.logging

def test_vid(session, m, vid_file, verbose=True):
  assert m.batch_size == 1
  tracks = tpn_test_iterator(vid_file)
  # import pdb
  # pdb.set_trace()
  cum_acc_static = 0.
  cum_acc_lstm = 0.
  log.info(vid_file)
  vid_res = []
  for ind, track in enumerate(tracks, start=1):
    # process track data
    track_length = track['feature'].shape[0]
    expend_feat = np.zeros((m.num_steps,) + track['feature'].shape[1:])
    expend_feat[:track_length] = track['feature']

    # extract features
    state = session.run([m.initial_state])
    cls_scores, bbox_deltas, end_probs, state = session.run(
    [m.cls_scores, m.bbox_pred, m.end_probs, m.final_state],
     {m.input_data: expend_feat[np.newaxis,:,:],
      m.initial_state: state[0]})

    # process outputs
    cls_labels = track['class_label']
    gt_len = cls_labels.shape[0]
    bbox_pred = bbox_transform_inv(track['roi'], bbox_deltas[:gt_len,:])
    cls_pred_lstm = np.argmax(cls_scores, axis=1)[:gt_len]
    end_probs = end_probs[:gt_len]

    # calculate accuracy comparison
    cls_pred_static = np.argmax(track['scores'], axis=1)[:gt_len]
    cum_acc_lstm += np.mean((cls_labels == cls_pred_lstm))
    cum_acc_static += np.mean((cls_labels == cls_pred_static))

    # save outputs
    track_res = {}
    for key in ['roi', 'frame', 'bbox', 'scores', 'anchor']:
      track_res[key] = track[key]
    track_res['scores_lstm'] = cls_scores[:gt_len,:]
    track_res['end_lstm'] = end_probs
    track_res['bbox_lstm'] = bbox_pred.reshape((gt_len, -1, 4))
    vid_res.append(track_res)
  cum_acc_lstm /= len(tracks)
  cum_acc_static /= len(tracks)
  log.info("Accuracy (Static): {:.03f} Accuracy (LSTM): {:.03f}".format(cum_acc_static, cum_acc_lstm))
  return vid_res

class TestConfig(object):
  """Default config."""
  init_scale = 0.01
  learning_rate = 0.001
  momentum = 0.9
  max_grad_norm = 1.5
  num_steps = 20
  input_size = 1024
  hidden_size = 1024
  max_epoch = 5
  iter_epoch = 2000
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 1
  num_classes = 31
  cls_weight = 1.0
  bbox_weight = 0.0
  ending_weight = 1.0
  vid_per_batch = 4
  cls_init = ''
  bbox_init = ''

def main(args):
  if not args.data_path:
    raise ValueError("Must set --data_path to TPN data directory")

  log.info("Processing data...")
  # raw_data = tpn_raw_data(args.data_path)
  # train_data, valid_data = raw_data

  config = TestConfig()
  config.num_layers = args.num_layers
  config.type = args.type
  config.hidden_size = config.input_size = args.input_size

  #tf.set_random_seed(1017)
  vids = glob.glob(osp.join(args.data_path, '*'))
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale, seed=1017)
    with tf.variable_scope("model", reuse=None, initializer=None):
      m = TPNModel(is_training=False, config=config)

    # restoring variables
    saver = tf.train.Saver()
    log.info("Retoring from {}".format(args.model_path))
    saver.restore(session, args.model_path)
    for vid_file in vids:
      vid_name = osp.split(vid_file)[-1]
      save_dir = osp.join(args.save_dir, vid_name)
      if not osp.isdir(save_dir):
        os.makedirs(save_dir)
      outputs = test_vid(session, m, vid_file, verbose=True)
      for track_id, track in enumerate(outputs):
        with open(osp.join(save_dir, '{:06d}.pkl'.format(track_id)), 'wb') as f:
          cPickle.dump(track, f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Extracting recurrent features.')
  parser.add_argument('data_path',
      help='Data path')
  parser.add_argument('save_dir',
      help='Result directory')
  parser.add_argument('model_path', help='model_stored path')
  parser.add_argument('num_layers', type=int,
      help='Number of layers')
  parser.add_argument('input_size', type=int,
      help='Input size.')
  parser.add_argument('--type', type=str, choices=['residual', 'basic'], default='residual',
      help='Type of LSTM cells. [residual]')
  args = parser.parse_args()

  main(args)
