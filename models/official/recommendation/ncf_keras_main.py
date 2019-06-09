# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NCF framework to train and evaluate the NeuMF model.

The NeuMF model assembles both MF and MLP models under the NCF framework. Check
`neumf_model.py` for more details about the models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.datasets import movielens
from official.recommendation import constants as rconst
from official.recommendation import ncf_common
from official.recommendation import neumf_model
from official.utils.logs import logger
from official.utils.logs import mlperf_helper
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils
from official.utils.misc import model_helpers


FLAGS = flags.FLAGS


def _keras_loss(y_true, y_pred):
  # Here we are using the exact same loss used by the estimator
  loss = tf.keras.losses.sparse_categorical_crossentropy(
      y_pred=y_pred,
      y_true=tf.cast(y_true, tf.int32),
      from_logits=True)
  return loss


def _get_metric_fn(params):
  """Get the metrix fn used by model compile."""
  batch_size = params["batch_size"]

  def metric_fn(y_true, y_pred):
    """Returns the in_top_k metric."""
    softmax_logits = y_pred[0, :]
    logits = tf.slice(softmax_logits, [0, 1], [batch_size, 1])

    # The dup mask should be obtained from input data, but we did not yet find
    # a good way of getting it with keras, so we set it to zeros to neglect the
    # repetition correction
    dup_mask = tf.zeros([batch_size, 1])

    _, _, in_top_k, _, _ = (
        neumf_model.compute_eval_loss_and_metrics_helper(
            logits,
            softmax_logits,
            dup_mask,
            params["num_neg"],
            params["match_mlperf"],
            params["use_xla_for_gpu"]))

    is_training = tf.keras.backend.learning_phase()
    if isinstance(is_training, int):
      is_training = tf.constant(bool(is_training), dtype=tf.bool)

    in_top_k = tf.cond(
        is_training,
        lambda: tf.zeros(shape=in_top_k.shape, dtype=in_top_k.dtype),
        lambda: in_top_k)

    return in_top_k

  return metric_fn


def _get_train_and_eval_data(producer, params):
  """Returns the datasets for training and evalutating."""

  def preprocess_train_input(features, labels):
    """Pre-process the training data.

    This is needed because:
    - Distributed training does not support extra inputs. The current
      implementation does not use the VALID_POINT_MASK in the input, which makes
      it extra, so it needs to be removed.
    - The label needs to be extended to be used in the loss fn
    """
    features.pop(rconst.VALID_POINT_MASK)
    labels = tf.expand_dims(labels, -1)
    return features, labels

  train_input_fn = producer.make_input_fn(is_training=True)
  train_input_dataset = train_input_fn(params).map(
      preprocess_train_input)

  def preprocess_eval_input(features):
    """Pre-process the eval data.

    This is needed because:
    - Distributed training does not support extra inputs. The current
      implementation does not use the DUPLICATE_MASK in the input, which makes
      it extra, so it needs to be removed.
    - The label needs to be extended to be used in the loss fn
    """
    features.pop(rconst.DUPLICATE_MASK)
    labels = tf.zeros_like(features[movielens.USER_COLUMN])
    labels = tf.expand_dims(labels, -1)
    return features, labels

  eval_input_fn = producer.make_input_fn(is_training=False)
  eval_input_dataset = eval_input_fn(params).map(
      lambda features: preprocess_eval_input(features))

  return train_input_dataset, eval_input_dataset


class IncrementEpochCallback(tf.keras.callbacks.Callback):
  """A callback to increase the requested epoch for the data producer.

  The reason why we need this is because we can only buffer a limited amount of
  data. So we keep a moving window to represent the buffer. This is to move the
  one of the window's boundaries for each epoch.
  """

  def __init__(self, producer):
    self._producer = producer

  def on_epoch_begin(self, epoch, logs=None):
    self._producer.increment_request_epoch()


def _get_keras_model(params):
  """Constructs and returns the model."""
  batch_size = params['batch_size']

  # The input layers are of shape (1, batch_size), to match the size of the
  # input data. The first dimension is needed because the input data are
  # required to be batched to use distribution strategies, and in this case, it
  # is designed to be of batch_size 1 for each replica.
  user_input = tf.keras.layers.Input(
      shape=(batch_size,),
      batch_size=params["batches_per_step"],
      name=movielens.USER_COLUMN,
      dtype=tf.int32)

  item_input = tf.keras.layers.Input(
      shape=(batch_size,),
      batch_size=params["batches_per_step"],
      name=movielens.ITEM_COLUMN,
      dtype=tf.int32)

  base_model = neumf_model.construct_model(
      user_input, item_input, params, need_strip=True)

  base_model_output = base_model.output

  logits = tf.keras.layers.Lambda(
      lambda x: tf.expand_dims(x, 0),
      name="logits")(base_model_output)

  zeros = tf.keras.layers.Lambda(
      lambda x: x * 0)(logits)

  softmax_logits = tf.keras.layers.concatenate(
      [zeros, logits],
      axis=-1)

  keras_model = tf.keras.Model(
      inputs=[user_input, item_input],
      outputs=softmax_logits)

  keras_model.summary()
  return keras_model


def run_ncf(_):
  """Run NCF training and eval with Keras."""
  # TODO(seemuch): Support different train and eval batch sizes
  if FLAGS.eval_batch_size != FLAGS.batch_size:
    logging.warning(
        "The Keras implementation of NCF currently does not support batch_size "
        "!= eval_batch_size ({} vs. {}). Overriding eval_batch_size to match "
        "batch_size".format(FLAGS.eval_batch_size, FLAGS.batch_size)
        )
    FLAGS.eval_batch_size = FLAGS.batch_size

  params = ncf_common.parse_flags(FLAGS)
  batch_size = params["batch_size"]

  # ncf_common rounds eval_batch_size (this is needed due to a reshape during
  # eval). This carries over that rounding to batch_size as well.
  params['batch_size'] = params['eval_batch_size']

  num_users, num_items, num_train_steps, num_eval_steps, producer = (
      ncf_common.get_inputs(params))

  params["num_users"], params["num_items"] = num_users, num_items
  producer.start()
  model_helpers.apply_clean(flags.FLAGS)

  batches_per_step = params["batches_per_step"]
  train_input_dataset, eval_input_dataset = _get_train_and_eval_data(producer,
                                                                     params)
  # It is required that for distributed training, the dataset must call
  # batch(). The parameter of batch() here is the number of replicas involed,
  # such that each replica evenly gets a slice of data.
  train_input_dataset = train_input_dataset.batch(batches_per_step)
  eval_input_dataset = eval_input_dataset.batch(batches_per_step)

  strategy = ncf_common.get_distribution_strategy(params)
  with distribution_utils.get_strategy_scope(strategy):
    keras_model = _get_keras_model(params)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params["learning_rate"],
        beta_1=params["beta1"],
        beta_2=params["beta2"],
        epsilon=params["epsilon"])
    time_callback = keras_utils.TimeHistory(batch_size, FLAGS.log_steps)

    keras_model.compile(
        loss=_keras_loss,
        metrics=[_get_metric_fn(params)],
        optimizer=optimizer,
        cloning=params["clone_model_in_keras_dist_strat"])

    history = keras_model.fit(train_input_dataset,
                              epochs=FLAGS.train_epochs,
                              callbacks=[
                                  IncrementEpochCallback(producer),
                                  time_callback],
                              verbose=2)

    logging.info("Training done. Start evaluating")

    eval_results = keras_model.evaluate(
        eval_input_dataset,
        steps=num_eval_steps,
        verbose=2)

  logging.info("Keras evaluation is done.")

  stats = build_stats(history, eval_results, time_callback)
  return stats


def build_stats(history, eval_result, time_callback):
  """Normalizes and returns dictionary of stats.

    Args:
      history: Results of the training step. Supports both categorical_accuracy
        and sparse_categorical_accuracy.
      eval_output: Output of the eval step. Assumes first value is eval_loss and
        second value is accuracy_top_1.
      time_callback: Time tracking callback likely used during keras.fit.
    Returns:
      Dictionary of normalized results.
  """
  stats = {}
  if history and history.history:
    train_history = history.history
    stats['loss'] = train_history['loss'][-1]

  if eval_result:
    stats['eval_loss'] = eval_result[0]
    stats['eval_hit_rate'] = eval_result[1]

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats['step_timestamp_log'] = timestamp_log
    stats['train_finish_time'] = time_callback.train_finish_time
    if len(timestamp_log) > 1:
      stats['avg_exp_per_second'] = (
          time_callback.batch_size * time_callback.log_steps *
          (len(time_callback.timestamp_log)-1) /
          (timestamp_log[-1].timestamp - timestamp_log[0].timestamp))

  return stats


def main(_):
  with logger.benchmark_context(FLAGS), \
      mlperf_helper.LOGGER(FLAGS.output_ml_perf_compliance_logging):
    mlperf_helper.set_ncf_root(os.path.split(os.path.abspath(__file__))[0])
    if FLAGS.tpu:
      raise ValueError("NCF in Keras does not support TPU for now")
    run_ncf(FLAGS)


if __name__ == "__main__":
  ncf_common.define_ncf_flags()
  absl_app.run(main)
