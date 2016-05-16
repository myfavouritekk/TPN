# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import random
#random.seed(1017)

import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
from rnn_cells import TPNLSTMCell, ResLSTMCell
from data_io import tpn_iterator, tpn_raw_data
import cPickle
import os
import os.path as osp
import glog as log

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

class TPNModel(object):
  """The TPN model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.input_size = input_size = config.input_size
    self.num_classes = num_classes = config.num_classes
    self.vid_per_batch = config.vid_per_batch
    size = config.hidden_size
    self.cls_weight = config.cls_weight
    self.bbox_weight = config.bbox_weight
    self.ending_weight = config.ending_weight
    self.iter_epoch = config.iter_epoch
    self.momentum = config.momentum

    # placeholders for inputs and outputs
    self._input_data = inputs = tf.placeholder(tf.float32, [batch_size, num_steps, input_size])
    self._cls_targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._bbox_targets = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes * 4])
    self._bbox_weights = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes * 4])
    self._end_targets = tf.placeholder(tf.float32, [batch_size, num_steps])

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # original inputs: batch_size * input_size * num_steps
    # after process: num_steps * [batch_size, input_size]
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]


    lstm_cell = ResLSTMCell(size)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    # TODO: decide initial state
    self._initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    # output: (num_steps * batch_size) * input_size
    output = tf.reshape(tf.concat(0, outputs), [-1, size])

    # build losses
    # class score
    if config.cls_init:
      # use pre-trained weights to initilize
      with open(config.cls_init, 'rb') as f:
        log.info("Loading classificiation params from {}".format(config.cls_init))
        cls_w, cls_b = cPickle.load(f)
        softmax_w = tf.get_variable("softmax_w", initializer=tf.constant(cls_w))
        softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(cls_b))
    else:
      softmax_w = tf.get_variable("softmax_w")
      softmax_b = tf.get_variable("softmax_b", [num_classes], initializer=tf.constant_initializer(0.))
    logits = tf.matmul(output, softmax_w) + softmax_b
    # transpose cls_targets to make num_steps the leading axis
    cls_targets = tf.reshape(tf.transpose(self._cls_targets), [-1])
    loss_cls_score = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, cls_targets, name='loss_cls_score')
    self._cls_cost = cls_cost = tf.reduce_sum(loss_cls_score) / batch_size / num_steps

    # boudning box regression: L2 loss
    if config.bbox_init:
      with open(config.bbox_init, 'rb') as f:
        log.info("Loading bbox regression params from {}".format(config.bbox_init))
        bbox_w, bbox_b = cPickle.load(f)
      bbox_w = tf.get_variable("bbox_w", initializer=tf.constant(bbox_w))
      bbox_b = tf.get_variable("bbox_b", initializer=tf.constant(bbox_b))
    else:
      bbox_w = tf.get_variable("bbox_w", [size, num_classes * 4])
      bbox_b = tf.get_variable("bbox_b", [num_classes * 4])
    bbox_pred = tf.matmul(output, bbox_w) + bbox_b
    # permute num_steps and batch_size
    bbox_targets = tf.reshape(tf.transpose(self._bbox_targets, (1, 0, 2)), [-1, 4 * num_classes])
    self._bbox_cost = bbox_cost = tf.nn.l2_loss(bbox_pred - bbox_targets) / batch_size / num_steps / 4.
    #self._bbox_cost = bbox_cost = tf.constant(0.)

    # ending signal
    end_w = tf.get_variable("end_w", [size, 1])
    end_b = tf.get_variable("end_b", [1], initializer=tf.constant_initializer(0.))
    end_pred = tf.matmul(output, end_w) + end_b
    end_targets = tf.reshape(tf.transpose(self._end_targets), [-1, 1])
    loss_ending = tf.nn.sigmoid_cross_entropy_with_logits(end_pred, end_targets, name='loss_ending')
    self._end_cost = end_cost = tf.reduce_sum(loss_ending) / batch_size / num_steps

    self._cost = cost = cls_cost * self.cls_weight + bbox_cost * self.bbox_weight + end_cost * self.ending_weight
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))
    self.global_norm = global_norm

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def cls_targets(self):
    return self._cls_targets

  @property
  def bbox_targets(self):
    return self._bbox_targets

  @property
  def bbox_weights(self):
    return self._bbox_weights

  @property
  def end_targets(self):
    return self._end_targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def cls_cost(self):
    return self._cls_cost

  @property
  def bbox_cost(self):
    return self._bbox_cost

  @property
  def end_cost(self):
    return self._end_cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


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

