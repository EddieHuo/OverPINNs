import os

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax
jax.config.update("jax_default_matmul_precision", "highest")
# jax.config.update('jax_array', True)  # 正确配置项为jax_array
# jax.config.update("jax_debug_nans", True)  # 调试配置保留
# jax.config.update('jax_parallel_functions_output_gda', True)
# jax.config.update('jax_spmd_mode', 'allow_all')
from jax import devices
from jax.experimental import mesh_utils
from jax.sharding import Mesh

# 查看各GPU显存状态
for dev in devices():
    stats = dev.memory_stats()
    # Handle devices without memory stats
    allocated = stats.get('bytes_in_use', 0) / 1e9 if stats else 0
    total = stats.get('total_bytes', 0) / 1e9 if stats else 0
    print(f"GPU{dev.id}: Allocated {allocated:.2f}GB / {total:.2f}GB")

import train
import eval

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/sota.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):

    

    # # 在main函数开始处添加
    # if jax.process_count() > 1:
    #     if FLAGS.config.training.num_nodes != jax.process_count():
    #         raise ValueError(f"Number of processes ({jax.process_count()}) must match num_nodes ({FLAGS.config.training.num_nodes})")

    # # 创建设备Mesh
    # devices = mesh_utils.create_device_mesh(
    #     (FLAGS.config.training.num_nodes, FLAGS.config.training.gpus_per_node),
    #     axis_names=('nodes', 'gpus')
    # )
    # mesh = Mesh(devices, ('nodes', 'gpus'))
    

    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

        eval.evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)

    


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
