#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn
from rnn_cells import TPNLSTMCell, ResLSTMCell
import glog as log
import cPickle

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


    self.type = config.type
    if self.type == 'residual':
      lstm_cell = ResLSTMCell(size)
    elif self.type == 'basic':
      lstm_cell = tf.models.rnn.rnn_cell.BasicLSTMCell(size)
    else:
      raise ValueError('Unknown LSTM cell type: {}.'.format(self.type))
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
      softmax_w = tf.get_variable("softmax_w", [size, num_classes])
      softmax_b = tf.get_variable("softmax_b", [num_classes], initializer=tf.constant_initializer(0.))
    logits = tf.matmul(output, softmax_w) + softmax_b
    self._cls_scores = tf.nn.softmax(logits, name='cls_scores')
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
    self._bbox_pred = bbox_pred = tf.matmul(output, bbox_w) + bbox_b
    # permute num_steps and batch_size
    bbox_targets = tf.reshape(tf.transpose(self._bbox_targets, (1, 0, 2)), [-1, 4 * num_classes])
    self._bbox_cost = bbox_cost = tf.nn.l2_loss(bbox_pred - bbox_targets) / batch_size / num_steps / 4.
    #self._bbox_cost = bbox_cost = tf.constant(0.)

    # ending signal
    end_w = tf.get_variable("end_w", [size, 1])
    end_b = tf.get_variable("end_b", [1], initializer=tf.constant_initializer(0.))
    end_pred = tf.matmul(output, end_w) + end_b
    end_targets = tf.reshape(tf.transpose(self._end_targets), [-1, 1])
    self._end_probs = tf.nn.sigmoid(end_pred, name='end_probs')
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

  @property
  def cls_scores(self):
    return self._cls_scores
  @property
  def bbox_pred(self):
    return self._bbox_pred
  @property
  def end_probs(self):
    return self._end_probs


class BiTPNModel(object):
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


    self.type = config.type
    if self.type == 'residual':
      raise
      lstm_cell = ResLSTMCell(size)
    elif self.type == 'basic':
      # forward and backward cell
      with tf.variable_scope("forward"):
        f_lstm_cell = tf.models.rnn.rnn_cell.BasicLSTMCell(size)
      with tf.variable_scope("backward"):
        b_lstm_cell = tf.models.rnn.rnn_cell.BasicLSTMCell(size)
    else:
      raise ValueError('Unknown LSTM cell type: {}.'.format(self.type))
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    with tf.variable_scope("forward"):
      f_cell = tf.nn.rnn_cell.MultiRNNCell([f_lstm_cell] * config.num_layers)
    with tf.variable_scope("backward"):
      b_cell = tf.nn.rnn_cell.MultiRNNCell([b_lstm_cell] * config.num_layers)

    # TODO: decide initial state
    self._initial_state = f_cell.zero_state(batch_size, tf.float32)
    self._b_initial_state = b_cell.zero_state(batch_size, tf.float32)

    with tf.variable_scope("forward"):
      f_outputs, f_state = rnn.rnn(f_cell, inputs, initial_state=self._initial_state)
    with tf.variable_scope("backward"):
      b_outputs, b_state = rnn.rnn(b_cell, inputs[::-1], initial_state=self._b_initial_state)
    b_outputs = b_outputs[::-1]

    # output: (num_steps * batch_size) * input_size
    f_output = tf.reshape(tf.concat(0, f_outputs), [-1, size])
    b_output = tf.reshape(tf.concat(0, b_outputs), [-1, size])

    if config.combine == 'max':
      output = tf.maximum(f_output, b_output)
    elif config.combine == 'ave':
      output = (f_output + b_output) / 2.
    elif config.combine == 'concat':
      output = tf.concat(1, [f_output, b_output])
      size *= 2
    else:
      raise NotImplementedError('Combine scheme {} not suported.'.foramt(config.combine))
    print output

    # build losses
    # class score
    if config.cls_init:
      # use pre-trained weights to initilize
      with open(config.cls_init, 'rb') as f:
        log.info("Loading classificiation params from {}".format(config.cls_init))
        cls_w, cls_b = cPickle.load(f)
        if config.combine == 'concat':
          cls_w = np.concatenate((cls_w / 2., cls_w / 2.), axis=0)
      softmax_w = tf.get_variable("softmax_w", initializer=tf.constant(cls_w))
      softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(cls_b))
    else:
      softmax_w = tf.get_variable("softmax_w", [size, num_classes])
      softmax_b = tf.get_variable("softmax_b", [num_classes], initializer=tf.constant_initializer(0.))
    logits = tf.matmul(output, softmax_w) + softmax_b
    self._cls_scores = tf.nn.softmax(logits, name='cls_scores')
    # transpose cls_targets to make num_steps the leading axis
    cls_targets = tf.reshape(tf.transpose(self._cls_targets), [-1])
    loss_cls_score = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, cls_targets, name='loss_cls_score')
    self._cls_cost = cls_cost = tf.reduce_sum(loss_cls_score) / batch_size / num_steps

    # boudning box regression: L2 loss
    if config.bbox_init:
      with open(config.bbox_init, 'rb') as f:
        log.info("Loading bbox regression params from {}".format(config.bbox_init))
        bbox_w, bbox_b = cPickle.load(f)
      if config.combine == 'concat':
        bbox_w = np.concatenate((bbox_w / 2., bbox_w / 2.), axis=0)
      bbox_w = tf.get_variable("bbox_w", initializer=tf.constant(bbox_w))
      bbox_b = tf.get_variable("bbox_b", initializer=tf.constant(bbox_b))
    else:
      bbox_w = tf.get_variable("bbox_w", [size, num_classes * 4])
      bbox_b = tf.get_variable("bbox_b", [num_classes * 4])
    self._bbox_pred = bbox_pred = tf.matmul(output, bbox_w) + bbox_b
    # permute num_steps and batch_size
    bbox_targets = tf.reshape(tf.transpose(self._bbox_targets, (1, 0, 2)), [-1, 4 * num_classes])
    self._bbox_cost = bbox_cost = tf.nn.l2_loss(bbox_pred - bbox_targets) / batch_size / num_steps / 4.
    #self._bbox_cost = bbox_cost = tf.constant(0.)

    # ending signal
    end_w = tf.get_variable("end_w", [size, 1])
    end_b = tf.get_variable("end_b", [1], initializer=tf.constant_initializer(0.))
    end_pred = tf.matmul(output, end_w) + end_b
    end_targets = tf.reshape(tf.transpose(self._end_targets), [-1, 1])
    self._end_probs = tf.nn.sigmoid(end_pred, name='end_probs')
    loss_ending = tf.nn.sigmoid_cross_entropy_with_logits(end_pred, end_targets, name='loss_ending')
    self._end_cost = end_cost = tf.reduce_sum(loss_ending) / batch_size / num_steps

    self._cost = cost = cls_cost * self.cls_weight + bbox_cost * self.bbox_weight + end_cost * self.ending_weight
    self._final_state = f_state
    self._b_final_state = b_state

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
  def initial_backward_state(self):
    return self._b_initial_state

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
  def final_backward_state(self):
    return self._b_final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def cls_scores(self):
    return self._cls_scores
  @property
  def bbox_pred(self):
    return self._bbox_pred
  @property
  def end_probs(self):
    return self._end_probs
