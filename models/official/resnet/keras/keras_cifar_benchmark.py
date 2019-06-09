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
"""Executes Keras benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import flags
import tensorflow as tf # pylint: disable=g-bad-import-order

from official.resnet import cifar10_main as cifar_main
from official.resnet.keras import keras_benchmark
from official.resnet.keras import keras_cifar_main
from official.resnet.keras import keras_common

MIN_TOP_1_ACCURACY = 0.925
MAX_TOP_1_ACCURACY = 0.938

FLAGS = flags.FLAGS
CIFAR_DATA_DIR_NAME = 'cifar-10-batches-bin'


class Resnet56KerasAccuracy(keras_benchmark.KerasBenchmark):
  """Accuracy tests for ResNet56 Keras CIFAR-10."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    """A benchmark class.

    Args:
      output_dir: directory where to output e.g. log files
      root_data_dir: directory under which to look for dataset
      **kwargs: arbitrary named arguments. This is needed to make the
                constructor forward compatible in case PerfZero provides more
                named arguments before updating the constructor.
    """

    self.data_dir = os.path.join(root_data_dir, CIFAR_DATA_DIR_NAME)
    flag_methods = [
        keras_common.define_keras_flags, cifar_main.define_cifar_flags
    ]

    super(Resnet56KerasAccuracy, self).__init__(
        output_dir=output_dir, flag_methods=flag_methods)

  def benchmark_graph_1_gpu(self):
    """Test keras based model with Keras fit and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    """Test keras based model with eager and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  def benchmark_2_gpu(self):
    """Test keras based model with eager and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu')
    FLAGS.dtype = 'fp32'
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  def benchmark_2_gpu_no_cloning(self):
    """Test keras based model with eager, distributed no-cloning."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu_no_cloning')
    FLAGS.dtype = 'fp32'
    FLAGS.clone_model_in_keras_dist_strat = False
    FLAGS.enable_eager = True
    self._run_and_report_benchmark()

  def benchmark_graph_2_gpu(self):
    """Test keras based model with Keras fit and distribution strategies."""
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_2_gpu')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_no_dist_strat(self):
    """Test keras based model with Keras fit but not distribution strategies."""
    self._setup()
    FLAGS.distribution_strategy = 'off'
    FLAGS.num_gpus = 1
    FLAGS.data_dir = self.data_dir
    FLAGS.batch_size = 128
    FLAGS.train_epochs = 182
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu_no_dist_strat')
    FLAGS.dtype = 'fp32'
    self._run_and_report_benchmark()

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = keras_cifar_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet56KerasAccuracy, self)._report_benchmark(
        stats,
        wall_time_sec,
        top_1_min=MIN_TOP_1_ACCURACY,
        top_1_max=MAX_TOP_1_ACCURACY,
        total_batch_size=FLAGS.batch_size,
        log_steps=100)


class Resnet56KerasBenchmarkBase(keras_benchmark.KerasBenchmark):
  """Short performance tests for ResNet56 via Keras and CIFAR-10."""

  def __init__(self, output_dir=None, default_flags=None):
    flag_methods = [
        keras_common.define_keras_flags, cifar_main.define_cifar_flags
    ]

    super(Resnet56KerasBenchmarkBase, self).__init__(
        output_dir=output_dir,
        flag_methods=flag_methods,
        default_flags=default_flags)

  def _run_and_report_benchmark(self):
    start_time_sec = time.time()
    stats = keras_cifar_main.run(FLAGS)
    wall_time_sec = time.time() - start_time_sec

    super(Resnet56KerasBenchmarkBase, self)._report_benchmark(
        stats,
        wall_time_sec,
        total_batch_size=FLAGS.batch_size,
        log_steps=FLAGS.log_steps)

  def benchmark_1_gpu_no_dist_strat(self):
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu_no_dist_strat(self):
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'off'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu_no_dist_strat')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_1_gpu(self):
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_graph_1_gpu(self):
    self._setup()
    FLAGS.num_gpus = 1
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
    FLAGS.batch_size = 128
    self._run_and_report_benchmark()

  def benchmark_2_gpu(self):
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu')
    FLAGS.batch_size = 128 * 2  # 2 GPUs
    self._run_and_report_benchmark()

  def benchmark_2_gpu_no_cloning(self):
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.enable_eager = True
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu_no_cloning')
    FLAGS.batch_size = 128 * 2  # 2 GPUs
    FLAGS.clone_model_in_keras_dist_strat = False
    self._run_and_report_benchmark()

  def benchmark_graph_2_gpu(self):
    self._setup()
    FLAGS.num_gpus = 2
    FLAGS.enable_eager = False
    FLAGS.distribution_strategy = 'default'
    FLAGS.model_dir = self._get_model_dir('benchmark_graph_2_gpu')
    FLAGS.batch_size = 128 * 2  # 2 GPUs
    self._run_and_report_benchmark()


class Resnet56KerasBenchmarkSynth(Resnet56KerasBenchmarkBase):
  """Synthetic benchmarks for ResNet56 and Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    default_flags = {}
    default_flags['skip_eval'] = True
    default_flags['use_synthetic_data'] = True
    default_flags['train_steps'] = 110
    default_flags['log_steps'] = 10

    super(Resnet56KerasBenchmarkSynth, self).__init__(
        output_dir=output_dir, default_flags=default_flags)


class Resnet56KerasBenchmarkReal(Resnet56KerasBenchmarkBase):
  """Real data benchmarks for ResNet56 and Keras."""

  def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
    default_flags = {}
    default_flags['skip_eval'] = True
    default_flags['data_dir'] = os.path.join(root_data_dir, CIFAR_DATA_DIR_NAME)
    default_flags['train_steps'] = 110
    default_flags['log_steps'] = 10

    super(Resnet56KerasBenchmarkReal, self).__init__(
        output_dir=output_dir, default_flags=default_flags)


if __name__ == '__main__':
  tf.test.main()
