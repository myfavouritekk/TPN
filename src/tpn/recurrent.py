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

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "default",
    "A type of model.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("log_path", '.', "log_path")
flags.DEFINE_string("save_path", '.', "save_path")
flags.DEFINE_integer("num_layers", 1, "number of LSTM layers")

FLAGS = flags.FLAGS

if not osp.isdir(FLAGS.save_path):
  os.makedirs(FLAGS.save_path)



class DefaultConfig(object):
  """Default config."""
  init_scale = 0.01
  learning_rate = 0.001
  momentum = 0.9
  max_grad_norm = 1.5
  num_layers = FLAGS.num_layers
  num_steps = 20
  input_size = 1024
  hidden_size = 1024
  max_epoch = 5
  iter_epoch = 2000
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 128
  num_classes = 31
  cls_weight = 1.0
  bbox_weight = 0.0
  ending_weight = 1.0
  vid_per_batch = 4
  cls_init = 'cls_score_vid_params.pkl'
  bbox_init = 'bbox_pred_vid_params.pkl'


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


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


def get_config():
  if FLAGS.model == "default":
    return DefaultConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to TPN data directory")

  log.info("Processing data...")
  raw_data = tpn_raw_data(FLAGS.data_path)
  train_data, valid_data = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

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
      save_path = osp.join(FLAGS.save_path, 'ResLSTM_{}'.format(FLAGS.num_layers))
      log.info("Save to {}_{}".format(save_path, (i+1)*m.iter_epoch))
      saver.save(session, save_path, global_step=(i+1) * m.iter_epoch)

if __name__ == "__main__":
  tf.app.run()

