import os
import time

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import os
import ml_collections
from absl import logging
# import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset
import yaml  # 导入 yaml 模块用于保存配置
import swanlab as wandb

# swanlab.sync_wandb()

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0, :]

    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Define residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Initialize model
    model = models.AllenCahn(config, u0, t_star, x_star)

    # Initialize evaluator
    evaluator = models.AllenCanhEvaluator(config, model)

    print("Waiting for JIT...")
    start_time = time.time()
    from jax import lax

    @jax.jit
    def cached_step(state, batch):
        return model.step(state, batch)

    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = cached_step(model.state, batch)  # 使用带缓存的jit

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # 定期保存检查点和配置文件
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                # 创建检查点目录
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                os.makedirs(ckpt_path, exist_ok=True)
                
                # 保存模型参数检查点
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
                print(f"模型检查点已保存到: {ckpt_path}")
                
                # 保存配置文件，确保后续预测时能加载相同的配置
                config_path = os.path.join(os.getcwd(), config.wandb.name, "config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config.to_dict(), f)
                print(f"配置文件已保存到: {config_path}")

    # 训练结束后确保最终模型被保存
    final_ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
    os.makedirs(final_ckpt_path, exist_ok=True)
    save_checkpoint(model.state, final_ckpt_path, keep=1)  # 只保留最终检查点
    
    # 再次保存配置文件以确保一致性
    final_config_path = os.path.join(os.getcwd(), config.wandb.name, "config.yaml")
    with open(final_config_path, 'w') as f:
        yaml.dump(config.to_dict(), f)
    
    print(f"\n训练完成！最终模型已保存到: {final_ckpt_path}")
    print(f"使用以下命令进行预测: python predict.py --model_dir={config.wandb.name} --save_dir=eval_results/{config.wandb.name}")

    return model
