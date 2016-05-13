# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
from rnn_cells import TPNLSTMCell
from data_io import tpn_iterator, tpn_raw_data

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "default",
    "A type of model.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


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

    # placeholders for inputs and outputs
    self._input_data = inputs = tf.placeholder(tf.float32, [batch_size, num_steps, input_size])
    self._cls_targets = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._bbox_targets = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes * 4])
    self._bbox_weights = tf.placeholder(tf.float32, [batch_size, num_steps, num_classes * 4])
    self._end_targets = tf.placeholder(tf.float32, [batch_size, num_steps])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.)
    # lstm_cell = TPNLSTMCell(size, forget_bias=1.)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    # TODO: decide initial state
    self._initial_state = cell.zero_state(batch_size, tf.float32)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # original inputs: batch_size * input_size * num_steps
    # after process: num_steps * [batch_size, input_size]
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]
    outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
    # output: (num_steps * batch_size) * input_size
    output = tf.reshape(tf.concat(1, outputs), [-1, size])

    # build losses
    # class score
    softmax_w = tf.get_variable("softmax_w", [size, num_classes])
    softmax_b = tf.get_variable("softmax_b", [num_classes], initializer=tf.constant_initializer(0.))
    logits = tf.matmul(output, softmax_w) + softmax_b
    cls_targets = tf.reshape(tf.transpose(self._cls_targets), [-1])
    loss_cls_score = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [cls_targets],
        [tf.ones([batch_size * num_steps])],
        name="loss_cls_score")
    self._cls_cost = cls_cost = tf.reduce_sum(loss_cls_score) / batch_size / num_steps

    # boudning box regression: SmoothL1Loss
    #  f(x) = 0.5 * x^2    if |x| < 1
    #         |x| - 0.5    otherwise
    bbox_w = tf.get_variable("bbox_w", [size, num_classes * 4])
    bbox_b = tf.get_variable("bbox_b", [num_classes * 4])
    bbox_pred = tf.matmul(output, bbox_w) + bbox_b
    # permute num_steps and batch_size
    bbox_targets = tf.reshape(tf.transpose(self._bbox_targets, (1, 0, 2)), [-1, 4 * num_classes])
    bbox_diff = tf.abs(tf.sub(bbox_pred, bbox_targets))
    less = tf.to_float(tf.less(bbox_diff, tf.constant(1.)))
    bbox_weights = tf.reshape(self._bbox_weights, [-1, 4 * num_classes])
    loss_bbox = tf.mul(tf.mul(less, tf.nn.l2_loss(bbox_diff)) + \
        tf.mul(1 - less, bbox_diff - tf.constant(0.5)), bbox_weights)
    self._bbox_cost = bbox_cost = tf.reduce_sum(loss_bbox) / batch_size / num_steps / 4.

    # ending signal
    end_w = tf.get_variable("end_w", [size, 1], initializer=tf.random_uniform_initializer(-0.001, 0.001))
    end_b = tf.get_variable("end_b", [1], initializer=tf.constant_initializer(0.))
    end_pred = tf.matmul(output, end_w) + end_b
    end_targets = tf.reshape(tf.transpose(self._end_targets), [-1, 1])
    # import pdb
    # pdb.set_trace()
    loss_ending = tf.nn.seq2seq.sequence_loss_by_example(
        [end_pred],
        [end_targets],
        [tf.ones([batch_size * num_steps])],
        softmax_loss_function=tf.nn.sigmoid_cross_entropy_with_logits,
        name="loss_ending")
    self._end_cost = end_cost = tf.reduce_sum(loss_ending) / batch_size / num_steps

    self._cost = cost = tf.add_n([cls_cost * self.cls_weight, bbox_cost * self.bbox_weight, end_cost * self.ending_weight])
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

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
  max_grad_norm = 5
  num_layers = 1
  num_steps = 20
  input_size = 1024
  hidden_size = 1024
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 128
  num_classes = 31
  cls_weight = 1.0
  bbox_weight = 0.0
  ending_weight = 0.0
  vid_per_batch = 4


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


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  # epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  cls_costs = 0.0
  bbox_costs = 0.0
  end_costs = 0.0
  display_iter = 100.
  state = m.initial_state.eval()
  for step in xrange(10000):
    x, cls_t, end_t, bbox_t, bbox_weights = tpn_iterator(data, m.batch_size, m.num_steps, m.num_classes, m.vid_per_batch)
    # import pdb
    # pdb.set_trace()
    cost, cls_cost, bbox_cost, end_cost, state, _ = session.run(
        [m.cost, m.cls_cost, m.bbox_cost, m.end_cost, m.final_state, eval_op],
         {m.input_data: x,
          m.cls_targets: cls_t,
          m.bbox_targets: bbox_t,
          m.bbox_weights: bbox_weights,
          m.end_targets: end_t,
          m.initial_state: state})
    costs += cost
    cls_costs += cls_cost
    bbox_costs += bbox_cost
    end_costs += end_cost

    if verbose and (step + 1) % display_iter == 0:
      costs /= display_iter
      cls_costs /= display_iter
      bbox_costs /= display_iter
      end_costs /= display_iter
      print "Iter {:06d}: cost {:.03f} = cls_cost {:.03f} * {:.02f} + end_cost {:.03f} * {:.02f} + bbox_cost {:.03f} * {:.02f}".format(
        step+1, costs, cls_costs, m.cls_weight,
                      end_costs, m.ending_weight,
                      bbox_costs, m.bbox_weight)
      costs = 0.0
      cls_costs = 0.0
      bbox_costs = 0.0
      end_costs = 0.0
      start_time = time.time()

  return costs


def get_config():
  if FLAGS.model == "default":
    return DefaultConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to TPN data directory")

  raw_data = tpn_raw_data(FLAGS.data_path)
  train_data, valid_data = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = TPNModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = TPNModel(is_training=False, config=config)

    tf.initialize_all_variables().run()

    for i in range(config.max_max_epoch):
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_cost = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Cost: %.3f" % (i + 1, train_cost))
      valid_cost = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Cost: %.3f" % (i + 1, valid_cost))

if __name__ == "__main__":
  tf.app.run()
