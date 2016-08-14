# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import random
#random.seed(1017)

import time

import numpy as np
import tensorflow as tf
from data_io import tpn_iterator, tpn_raw_data
import cPickle
import os
import os.path as osp
import glog as log
from model import TPNModel
import yaml
from easydict import EasyDict as edict

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("save_path", '.', "save_path")
flags.DEFINE_string("config", '.', "RNN config file.")
FLAGS = flags.FLAGS

if not osp.isdir(FLAGS.save_path):
  os.makedirs(FLAGS.save_path)

def run_epoch(session, m, data, eval_op, init_state, epoch_idx, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  tot_costs = 0.0
  costs = 0.0
  cls_costs = 0.0
  bbox_costs = 0.0
  end_costs = 0.0
  display_iter = 20.
  state = init_state
  for step in xrange(m.iter_epoch):
    x, cls_t, end_t, bbox_t, bbox_weights = tpn_iterator(data, m.batch_size, m.num_steps, m.num_classes, m.vid_per_batch)
    data_time = time.time()
    cost, cls_cost, bbox_cost, end_cost, state, _, global_norm = session.run(
        [m.cost, m.cls_cost, m.bbox_cost, m.end_cost, m.final_state, eval_op,  m.global_norm],
         {m.input_data: x,
          m.cls_targets: cls_t,
          m.bbox_targets: bbox_t,
          m.bbox_weights: bbox_weights,
          m.end_targets: end_t,
          m.initial_state: state})
    costs += cost
    tot_costs += cost
    cls_costs += cls_cost
    bbox_costs += bbox_cost
    end_costs += end_cost

    if verbose and (step + 1) % display_iter == 0:
      costs /= display_iter
      cls_costs /= display_iter
      bbox_costs /= display_iter
      end_costs /= display_iter
      log.info("Iter {:06d} {:.03f} s/iter data time: {:.03f} s: cost {:.03f} = cls_cost {:.03f} * {:.02f} + end_cost {:.03f} * {:.02f} + bbox_cost {:.03f} * {:.02f}. Global norm: {:.03f}".format(
        step+1+epoch_idx * m.iter_epoch,
        (time.time() - start_time) / display_iter,
        (data_time - start_time) / display_iter,
        costs, cls_costs, m.cls_weight,
        end_costs, m.ending_weight,
        bbox_costs, m.bbox_weight, global_norm))
      costs = 0.0
      cls_costs = 0.0
      bbox_costs = 0.0
      end_costs = 0.0
      start_time = time.time()

  return tot_costs / m.iter_epoch, state


def get_config(phase):
  config = {}
  raw_config = yaml.load(open(FLAGS.config).read())
  for section in ['init', 'model', phase]:
    assert section in raw_config
    for key in raw_config[section].keys():
      config[key] = raw_config[section][key]
  return edict(config)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to TPN data directory")

  log.info("Loading config file {}.".format(FLAGS.config))
  config = get_config('train')
  eval_config = get_config('test')

  log.info("Processing data...")
  raw_data = tpn_raw_data(FLAGS.data_path)
  train_data, valid_data = raw_data

  #tf.set_random_seed(1017)
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale, seed=1017)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = TPNModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = TPNModel(is_training=False, config=config)

    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    for i in range(config.max_epoch):
      lr_decay = config.lr_decay ** i
      m.assign_lr(session, config.learning_rate * lr_decay)
      
      if i == 0:
        state = m.initial_state.eval()
      log.info("Epoch: {} Learning rate {:.03e}.".format(i+1,session.run(m.lr)))
      train_cost, state = run_epoch(session, m, train_data, m.train_op,
                             state, i, verbose=True)
      log.info("Epoch: %d Train Cost: %.3f" % (i + 1, train_cost))
      save_path = osp.join(FLAGS.save_path, '{}LSTM_{}'.format(FLAGS.type, FLAGS.num_layers))
      log.info("Save to {}_{}".format(save_path, (i+1)*m.iter_epoch))
      saver.save(session, save_path, global_step=(i+1) * m.iter_epoch)

if __name__ == "__main__":
  tf.app.run()

