import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn.rnn_cell import *


class TPNLSTMCell(rnn.rnn_cell.RNNCell):
  """LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
    """
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._forget_bias = forget_bias

  @property
  def state_size(self):
    return 2 * self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "TPNLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = array_ops.split(1, 2, state)
      concat = linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * tf.nn.relu(j)
      new_h = tf.nn.relu(new_c) * sigmoid(o)

      return new_h, array_ops.concat(1, [new_c, new_h])

class ResLSTMCell(RNNCell):
  """Residual LSTM recurrent network cell.
    LSTM cell with identity mapping
  """

  def __init__(self, num_units, forget_bias=1.0):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
    """
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._forget_bias = forget_bias

  @property
  def state_size(self):
    return 2 * self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = array_ops.split(1, 2, state)
      concat = linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * tanh(j)
      # residual learning with identity mapping
      new_h = tanh(new_c) * sigmoid(o) + h

      return new_h, array_ops.concat(1, [new_c, new_h])
